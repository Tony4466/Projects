

#region ---------------- INITIALIZE BOARD ----------------
visuals_enabled = True
import pygame, sys, chess, chess.engine, opening_book, random, statistics, math, subprocess, stockfish
WIDTH = HEIGHT = 300#760 #size of board
SQ = WIDTH // 8
games = []
depth_log = []

#open window and load images
if __name__ == "__main__" and visuals_enabled:
    pygame.init()
    pygame.display.set_caption("Chess")
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    PIECE_IMAGES = {}
    for c in ['w', 'b']:
        for p in ['p','r','n','b','q','k']:
            PIECE_IMAGES[c+p] = pygame.transform.smoothscale(
            pygame.image.load(f"images/{c}{p}.png").convert_alpha(), (SQ, SQ)
       )

def draw():
    for r in range(8):
        for c in range(8):
            col = WHITE if (r+c)%2==0 else BLACK
            if selected and (r,c) in legal_moves:
                col = HIGHLIGHT1 if (r+c)%2==0 else HIGHLIGHT2
            pygame.draw.rect(WIN, col, (c*SQ, r*SQ, SQ, SQ))
            p = board["grid"][r][c]
            if p != '.':
                key = ('w' if p.isupper() else 'b') + p.lower()
                WIN.blit(PIECE_IMAGES[key], (c*SQ, r*SQ))
    pygame.display.update()
def opponent(color): return "black" if color=="white" else "white"
def sq(r, c): return r*8 + c
#endregion

#region ---------------- PLAYERS ----------------
HUMAN = 2
MY_AI = 1
STOCKFISH = 0

WHITE_PLAYER = STOCKFISH
BLACK_PLAYER = MY_AI

# -----player options-----
STOCKFISH_PATH = "engines/stockfish/stockfish-macos-m1-apple-silicon"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({
    #"Skill Level": 0,          # Made to be more human like, less consistent elo
    "UCI_LimitStrength": True,  # Enable strength limiting
    "UCI_Elo": 1800             # Limits elo by search depth, plays best move it saw, more consistent elo
})
"""
Skill 0: ~800 Elo
Skill 5: ~1300 Elo
Skill 10: ~1800 Elo
Skill 15: ~2200 Elo
Skill 20: ~2850+ Elo
"""
stockfish_delay = 0
stockfish_time = 1
show_stockfish_eval = True   #prints every turn
show_my_ai_eval = False   #prints every My AI turn
prev_evl = 0   #used for blunder detection


killers_active = True
history_active = True
q_checks = False
my_eval = []
stockfish_eval = []
use_opening_book = True
opening_response_delay = 40
response_time = 1000   #My AI think time
max_depth = 10
q_depth = 6
tie_score = -200
ASPIRATION_WINDOW = 40   #width of search from last evaluation
game_num = 1
cpl_white = []
cpl_black = []
time_log = []
total_time = 0

class SideStats:
    def __init__(self, color):
        self.color = color

        # --- Move-level ---
        self.moves = 0
        self.eval_sum = 0          # for ACPL
        self.eval_deltas = []      # raw eval change per move

        self.blunders = 0
        self.mistakes = 0
        self.inaccuracies = 0

        self.big_eval_moves = 0    # |eval| > 600
        self.forced_moves = 0

        # --- Search / time ---
        self.total_time = 0.0

        # --- Derived (filled later) ---
        self.accuracy = None
        self.chesscom_elo = None

    def record_move(self, eval_before, eval_after, forced=False):
        delta = abs(eval_after - eval_before)
        self.moves += 1
        self.eval_sum += delta
        self.eval_deltas.append(delta)

        if delta >= 300:
            self.blunders += 1
        elif delta >= 150:
            self.mistakes += 1
        elif delta >= 50:
            self.inaccuracies += 1

        if abs(eval_after) > 600:
            self.big_eval_moves += 1

        if forced:
            self.forced_moves += 1

    def finalize(self):
        self.acpl = statistics.mean([a if a<1000 else 1000 for a in self.eval_deltas])
        self.big_eval_pct = self.big_eval_moves / max(1, self.moves)
        self.avg_time = round(statistics.mean(time_log)/1000,3) if time_log else 0

class GameStats:
    def __init__(self, game_num, engine_config):
        self.game_num = game_num

        # Engine flags
        self.killers_active = engine_config.get("killers_active", False)
        self.history_active = engine_config.get("history_active", False)
        self.opening_response_delay = engine_config.get("opening_delay", 0)

        self.ASPIRATION_WINDOW = engine_config.get("ASPIRATION_WINDOW", None)
        self.tie_score = engine_config.get("tie_score", None)
        self.q_checks = engine_config.get("q_checks", False)

        # Per-side stats
        self.white = SideStats("white")
        self.black = SideStats("black")

        # External analysis
        self.stockfish_eval = {"white": None, "black": None}

    def record_move(self, color, **kwargs):
        if color == "white":
            self.white.record_move(**kwargs)
        else:
            self.black.record_move(**kwargs)

    def finalize(self):
        self.moves = stock_move_history(board)
        self.opening = board["opening_name"]
        self.white.finalize()
        self.black.finalize()

    def to_excel_rows(self):
        rows = []
        for side in [self.white, self.black]:
            rows.append({
                "game_num": self.game_num,
                "color": side.color,

                "killers_active": self.killers_active,
                "history_active": self.history_active,
                "opening_response_delay": self.opening_response_delay,

                "moves": side.moves,
                "acpl": round(side.acpl, 2),
                "blunders": side.blunders,
                "mistakes": side.mistakes,
                "inaccuracies": side.inaccuracies,
                "big_eval_pct": round(side.big_eval_pct, 3),

                "avg_time": round(side.avg_time, 3),
                "max_depth": max_depth,
                "q_depth": q_depth,

                "ASPIRATION_WINDOW": ASPIRATION_WINDOW,
                "tie_score": tie_score,

                "moves": self.moves

            })

        return rows

stats = GameStats(
    game_num=game_num,
    engine_config={
        "killers_active": killers_active,
        "history_active": history_active,
        "opening_delay": opening_response_delay,
        "ASPIRATION_WINDOW": ASPIRATION_WINDOW,
        "tie_score": tie_score,
        "q_depth": q_depth
    }
)


#endregion

#region ---------------- COLORS ----------------
WHITE = (240, 248, 255)
BLACK = (95, 158, 160)
HIGHLIGHT1 = (135, 206, 235)
HIGHLIGHT2 = (10, 129, 140)
#endregion

#region ---------------- MOVE LEGALITY ----------------
def make_move(bd, sr, sc, er, ec):
    g = bd["grid"]
    piece = g[sr][sc]
    captured = g[er][ec]

    if bd["en_passant"] is not None:
        _, old_file = bd["en_passant"]
        bd["zobrist"] ^= Z_EP[old_file]

    color = "white" if piece.isupper() else "black"
    opp = opponent(color)
    pt = piece.lower()

    # --- create undo ---
    undo = Undo(
        sr, sc, er, ec, captured,
        bd["en_passant"],
        bd["castling"].copy(),
        False
    )
    # --- snapshot incremental state ---
    undo.zobrist = bd["zobrist"]
    undo.material = bd["material"]
    undo.pst = bd["pst"]
    undo.king_pos = bd["king_pos"].copy()
    undo.phase = bd["phase"]
    undo.pawn_hash = bd["pawn_hash"]
    undo.halfmove_clock = bd["halfmove_clock"]
    undo.opening_name = bd["opening_name"]
    undo.eco = bd["eco"]

    for k, v in bd["castling"].items():
        if v:
            bd["zobrist"] ^= Z_CASTLE[k]

    # =====================================================
    # 1. REMOVE MOVING PIECE FROM SOURCE
    # =====================================================
    g[sr][sc] = '.'
    bd["zobrist"] ^= Z_PIECES[(sr, sc, piece)]
    bd["pst"] -= pst_value(pt, sr, sc, color, bd)
    bd["pieces"][color][pt].remove((sr, sc))
    if pt == 'p':
        bd["pawn_hash"] ^= Z_PIECES[(sr, sc, piece)]

    # =====================================================
    # 2. NORMAL CAPTURE
    # =====================================================
    if captured != '.':
        cpt = captured.lower()
        bd["zobrist"] ^= Z_PIECES[(er, ec, captured)]
        bd["material"] += VALUES[cpt] if color == "white" else -VALUES[cpt]
        bd["pst"] -= pst_value(cpt, er, ec, opp, bd)
        bd["pieces"][opp][cpt].remove((er, ec))
        bd["phase"] -= PHASE_WEIGHTS[captured.lower()]

        if captured.lower() == 'r':
            back = 7 if opp == "white" else 0
            if er == back:
                if ec == 0:
                    bd["castling"][opp + "_queen"] = False
                elif ec == 7:
                    bd["castling"][opp + "_king"] = False

        if captured.lower() == 'p':
            bd["pawn_hash"] ^= Z_PIECES[(er, ec, captured)]


    # Reset halfmove clock on pawn move or capture
    if pt == 'p' or captured != '.':
        bd["halfmove_clock"] = 0
    else:
        bd["halfmove_clock"] += 1

    # =====================================================
    # 3. EN PASSANT CAPTURE
    # =====================================================
    undo.ep_capture = False
    if pt == 'p' and bd["en_passant"] == (er, ec):
        undo.ep_capture = True
        undo.ep_pawn_pos = (sr, ec)
        bd["pieces"][opp]['p'].remove((sr, ec))
        cap_piece = g[sr][ec]
        g[sr][ec] = '.'
        bd["zobrist"] ^= Z_PIECES[(sr, ec, cap_piece)]
        bd["pawn_hash"] ^= Z_PIECES[(sr, ec, cap_piece)]

        bd["material"] += 100 if color == "white" else -100
        bd["pst"] -= pst_value('p', sr, ec, opp, bd)

    # =====================================================
    # 4. PLACE MOVING PIECE
    # =====================================================
    g[er][ec] = piece
    bd["zobrist"] ^= Z_PIECES[(er, ec, piece)]
    bd["pst"] += pst_value(pt, er, ec, color, bd)
    bd["pieces"][color][pt].add((er, ec))
    if pt == 'p':
        bd["pawn_hash"] ^= Z_PIECES[(er, ec, piece)]

    # =====================================================
    # 5. PROMOTION
    # =====================================================
    if pt == 'p' and (er == 0 or er == 7):
        undo.promoted = True
        new_piece = 'Q' if color == "white" else 'q'

        # Remove pawn zobrist hash and add queen hash
        bd["zobrist"] ^= Z_PIECES[(er, ec, piece)]
        bd["zobrist"] ^= Z_PIECES[(er, ec, new_piece)]
        bd["pawn_hash"] ^= Z_PIECES[(er, ec, piece)]

        g[er][ec] = new_piece

        bd["material"] += 800 if color == "white" else -800

        # Replace pawn PST with queen PST (pawn PST was already added above)
        bd["pst"] -= pst_value('p', er, ec, color, bd)
        bd["pst"] += pst_value('q', er, ec, color, bd)

        bd["pieces"][color]['p'].remove((er, ec))
        bd["pieces"][color]['q'].add((er, ec))
        bd["phase"] += PHASE_WEIGHTS['q']

    # =====================================================
    # 6. KING POSITION UPDATE
    # =====================================================
    if pt == 'k':
        bd["king_pos"][color] = (er, ec)

    # =====================================================
    # 7. CASTLING ROOK MOVE
    # =====================================================
    undo.castle = False
    if pt == 'k' and abs(ec - sc) == 2:
        undo.castle = True

        if ec == 6:      # king side
            rf, rt = (er, 7), (er, 5)
        else:            # queen side
            rf, rt = (er, 0), (er, 3)

        undo.rook_from = rf
        undo.rook_to = rt

        rook = g[rf[0]][rf[1]]
        g[rf[0]][rf[1]] = '.'
        g[rt[0]][rt[1]] = rook

        bd["zobrist"] ^= Z_PIECES[(rf[0], rf[1], rook)]
        bd["zobrist"] ^= Z_PIECES[(rt[0], rt[1], rook)]

        bd["pieces"][color]['r'].remove(undo.rook_from)
        bd["pieces"][color]['r'].add(undo.rook_to)

    # =====================================================
    # 8. UPDATE CASTLING RIGHTS
    # =====================================================
    if pt == 'k':
        bd["castling"][color + "_king"] = False
        bd["castling"][color + "_queen"] = False

    if pt == 'r':
        back_rank = 7 if color == "white" else 0
        if sr == back_rank:
            if sc == 0:
                bd["castling"][color + "_queen"] = False
            elif sc == 7:
                bd["castling"][color + "_king"] = False

    for k, v in bd["castling"].items():
        if v:
            bd["zobrist"] ^= Z_CASTLE[k]

    # =====================================================
    # 9. UPDATE EN PASSANT SQUARE
    # =====================================================
    bd["en_passant"] = None
    if pt == 'p' and abs(er - sr) == 2:
        bd["en_passant"] = ((sr + er) // 2, sc)

    if bd["en_passant"] is not None:
        _, new_file = bd["en_passant"]
        bd["zobrist"] ^= Z_EP[new_file]

    bd["zobrist"] ^= Z_TURN
    POSITION_HISTORY.append(bd["zobrist"])

    return undo
def unmake_move(bd, undo):
    g = bd["grid"]

    piece = g[undo.er][undo.ec]
    color = "white" if piece.isupper() else "black"

    # ---------------- PROMOTION FIRST ----------------
    if undo.promoted:
        # Remove promoted queen
        bd["pieces"][color]['q'].remove((undo.er, undo.ec))
        piece = 'P' if color == "white" else 'p'
    else:
        pt = piece.lower()
        bd["pieces"][color][pt].remove((undo.er, undo.ec))

    # ---------------- RESTORE GRID ----------------
    g[undo.sr][undo.sc] = piece
    g[undo.er][undo.ec] = undo.captured
    bd["pieces"][color][piece.lower()].add((undo.sr, undo.sc))

    # ---------------- RESTORE CAPTURE ----------------
    if undo.captured != '.':
        opp = opponent(color)
        bd["pieces"][opp][undo.captured.lower()].add((undo.er, undo.ec))

    # ---------------- EN PASSANT ----------------
    if undo.ep_capture:
        r, c = undo.ep_pawn_pos
        pawn = 'p' if color == "white" else 'P'
        g[r][c] = pawn
        bd["pieces"][opponent(color)]['p'].add((r, c))

    # ---------------- CASTLING ----------------
    if undo.castle:
        rf, rt = undo.rook_from, undo.rook_to
        rook = g[rt[0]][rt[1]]
        g[rf[0]][rf[1]] = rook
        g[rt[0]][rt[1]] = '.'
        bd["pieces"][color]['r'].remove(rt)
        bd["pieces"][color]['r'].add(rf)

    # ---------------- RESTORE SNAPSHOTS ----------------
    POSITION_HISTORY.pop()
    bd["en_passant"] = undo.ep
    bd["castling"] = undo.castling
    bd["zobrist"] = undo.zobrist
    bd["material"] = undo.material
    bd["pst"] = undo.pst
    bd["king_pos"] = undo.king_pos
    bd["phase"] = undo.phase
    bd["pawn_hash"] = undo.pawn_hash
    bd["halfmove_clock"] = undo.halfmove_clock

    #rebuild_pieces(bd)
    #sanity(bd)

def pawn_moves(bd,r,c,color):
    g=bd["grid"]; ep=bd["en_passant"]
    d=-1 if color=="white" else 1
    start=6 if color=="white" else 1
    moves=[]
    if g[r+d][c]=='.':
        moves.append((r+d,c))
        if r==start and g[r+2*d][c]=='.':
            moves.append((r+2*d,c))
    for dc in (-1,1):
        nr,nc=r+d,c+dc
        if in_bounds(nr,nc):
            if g[nr][nc]!='.' and g[nr][nc].isupper()!=(color=="white"):
                moves.append((nr,nc))
            if ep==(nr,nc):
                moves.append((nr,nc))
    return moves
def knight_moves(bd,r,c,color):
    g=bd["grid"]; m=[]
    for dr,dc in [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2)]:
        nr,nc=r+dr,c+dc
        if in_bounds(nr,nc) and (g[nr][nc]=='.' or g[nr][nc].isupper()!=(color=="white")):
            m.append((nr,nc))
    return m
def slide_moves(bd,r,c,color,dirs):
    g=bd["grid"]; m=[]
    for dr,dc in dirs:
        nr,nc=r+dr,c+dc
        while in_bounds(nr,nc):
            if g[nr][nc]=='.': m.append((nr,nc))
            else:
                if g[nr][nc].isupper()!=(color=="white"): m.append((nr,nc))
                break
            nr+=dr; nc+=dc
    return m
def king_moves(bd, r, c, color):
    grid = bd["grid"]
    castle = bd["castling"]
    moves = []

    enemy = opponent(color)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc):
                    if grid[nr][nc] == '.' or grid[nr][nc].isupper() != (color == "white"):
                        # 🔴 CRITICAL LINE: king cannot move into attack
                        if not square_attacked(bd, nr, nc, enemy):
                            moves.append((nr, nc))

    # Castling logic (unchanged)
    if color == "white" and r == 7 and c == 4:
        if castle["white_king"] and grid[7][5] == grid[7][6] == '.' and not in_check(bd, "white"):
            if not square_attacked(bd, 7, 5, "black") and not square_attacked(bd, 7, 6, "black"):
                moves.append((7, 6))
        if castle["white_queen"] and grid[7][1] == grid[7][2] == grid[7][3] == '.' and not in_check(bd, "white"):
            if not square_attacked(bd, 7, 2, "black") and not square_attacked(bd, 7, 3, "black"):
                moves.append((7, 2))

    if color == "black" and r == 0 and c == 4:
        if castle["black_king"] and grid[0][5] == grid[0][6] == '.' and not in_check(bd, "black"):
            if not square_attacked(bd, 0, 5, "white") and not square_attacked(bd, 0, 6, "white"):
                moves.append((0, 6))
        if castle["black_queen"] and grid[0][1] == grid[0][2] == grid[0][3] == '.' and not in_check(bd, "black"):
            if not square_attacked(bd, 0, 2, "white") and not square_attacked(bd, 0, 3, "white"):
                moves.append((0, 2))

    return moves

def raw_moves(bd,r,c,color,ignore_castling=False):
    g=bd["grid"]; p=g[r][c]
    if p=='.': return []
    t=p.lower()
    if t=='p': return pawn_moves(bd,r,c,color)
    if t=='n': return knight_moves(bd,r,c,color)
    if t=='b': return slide_moves(bd,r,c,color,[(-1,-1),(-1,1),(1,-1),(1,1)])
    if t=='r': return slide_moves(bd,r,c,color,[(-1,0),(1,0),(0,-1),(0,1)])
    if t=='q': return slide_moves(bd,r,c,color,[(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)])
    if t=='k': return king_moves(bd,r,c,color) if not ignore_castling else []
    return []
def all_moves_for_board(bd, color):
    moves = []
    for piece_type, squares in bd["pieces"][color].items():
        for (r, c) in squares:
            for nr, nc in raw_moves(bd, r, c, color):
                undo = make_move(bd, r, c, nr, nc)
                if not in_check(bd, color):
                    moves.append((r,c,nr,nc))
                unmake_move(bd, undo)
    return moves

def square_attacked(bd, r, c, by_color):
    g = bd["grid"]
    is_white = (by_color == "white")

    # ---------------- PAWNS ----------------
    if is_white:
        pawn_sources = [(r + 1, c - 1), (r + 1, c + 1)]
    else:
        pawn_sources = [(r - 1, c - 1), (r - 1, c + 1)]

    for rr, cc in pawn_sources:
        if in_bounds(rr, cc):
            p = g[rr][cc]
            if p == ('P' if is_white else 'p'):
                return True

    # ---------------- KNIGHTS ----------------
    for dr, dc in [
        (-2,-1), (-2,1), (2,-1), (2,1),
        (-1,-2), (-1,2), (1,-2), (1,2)
    ]:
        rr, cc = r + dr, c + dc
        if in_bounds(rr, cc):
            p = g[rr][cc]
            if p == ('N' if is_white else 'n'):
                return True

    # ---------------- KING ----------------
    for dr, dc in [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),          (0,1),
        (1,-1),  (1,0),  (1,1)
    ]:
        rr, cc = r + dr, c + dc
        if in_bounds(rr, cc):
            p = g[rr][cc]
            if p == ('K' if is_white else 'k'):
                return True

    # ---------------- BISHOPS / QUEENS (DIAGONALS) ----------------
    for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc):
            p = g[rr][cc]
            if p != '.':
                if p.isupper() == is_white and p.lower() in ('b', 'q'):
                    return True
                break
            rr += dr
            cc += dc

    # ---------------- ROOKS / QUEENS (FILES & RANKS) ----------------
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc):
            p = g[rr][cc]
            if p != '.':
                if p.isupper() == is_white and p.lower() in ('r', 'q'):
                    return True
                break
            rr += dr
            cc += dc

    return False
def in_check(bd, color):
    kr, kc = bd["king_pos"][color]
    return square_attacked(bd, kr, kc, opponent(color))

def in_bounds(r,c): return 0 <= r < 8 and 0 <= c < 8
#endregion

#region ---------------- MOVE SEARCH ----------------
def get_move_for_turn(turn):
    if turn == "white":
        player = WHITE_PLAYER
    else:
        player = BLACK_PLAYER
    if player == HUMAN:
        return None
    elif player == MY_AI:
        if use_opening_book:
            entry = opening_book.OPENING_BOOK.get(board["zobrist"])
            if entry:
                pygame.time.delay(opening_response_delay)
                board["opening_name"] = entry["name"]
                board["eco"] = entry["eco"]
                sr,sc,er,ec = random.choices([a for a,_ in entry["moves"]],[a for _,a in entry["moves"]])[0]
                return sr,sc,er,ec
        info = iterative_deepening(board, turn, response_time)
        try:
            my_eval.append(-int(info[1]))
        except:
            my_eval.append(None)

        return info[0]

    elif player == STOCKFISH:
        pygame.time.delay(stockfish_delay)
        return stockfish_move(board, turn, stockfish_time)

def iterative_deepening(bd, turn, time_limit_ms):
    global depth_log
    TT.clear()

    start = pygame.time.get_ticks()
    last_score = 0
    depth = 1
    best_move = None
    best_score = -INF-10000
    mvs = all_moves_for_board(bd,turn)
    if len(mvs)==1:
        print("len 1")
        return mvs[0], evaluate(bd,turn)

    while True:
        alpha = last_score - ASPIRATION_WINDOW
        beta  = last_score + ASPIRATION_WINDOW
        fails = 0

        while True:
            score, move = search_depth_window(bd, depth, turn, alpha, beta)

            if score >= INF - 100:
                return move, score

            if alpha < score < beta:
                last_score = score
                if move:
                    best_move = move
                    best_score = score
                break

            fails += 1

            # 🔥 HARD FAILSAFE
            if fails >= 1:
                alpha = -INF
                beta = INF
                score, move = search_depth_window(bd, depth, turn, alpha, beta)
                if move:
                    best_move = move
                    best_score = score
                last_score = score
                break

            if score <= alpha:
                alpha -= ASPIRATION_WINDOW * 4
            else:
                beta += ASPIRATION_WINDOW * 4

        # ⏱ Time check
        if pygame.time.get_ticks() - start > time_limit_ms or depth == max_depth:
            depth_log.append(depth)
            time_log.append(pygame.time.get_ticks() - start)
            break
        depth += 1

    return best_move, best_score
def search_depth_window(bd, depth, turn, alpha, beta):
    moves = all_moves_for_board(bd, turn)
    moves = order_moves(bd, moves, turn)
    move = None
    if not moves:
        if in_check(bd, turn):
            return -INF, []
        return 0, []

    best = -INF
    for sr, sc, er, ec in moves:
        undo = make_move(bd, sr, sc, er, ec)
        score = -alphabeta(bd, opponent(turn), depth - 1, -beta, -alpha, 1)
        unmake_move(bd, undo)

        if score > best:
            best = score
            move = (sr, sc, er, ec)

        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best, move
def alphabeta(bd, color, depth, alpha, beta, ply=0):
    h = bd["zobrist"]
    orig_alpha = alpha

    tt_move = None
    if h in TT:
        tt_score, tt_depth, tt_flag, tt_move = TT[h]
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_score
            elif tt_flag == LOWER:
                alpha = max(alpha, tt_score)
            elif tt_flag == UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    if is_repetition() or fifty_move_draw(bd):
        return tie_score

    moves = all_moves_for_board(bd, color)

    if not moves:
        if in_check(bd, color):
            return -INF + ply
        return tie_score

    if depth == 0:
        return quiescence(bd, color, alpha, beta)

    best = -INF
    next_turn = opponent(color)
    best_move = None

    for sr, sc, er, ec in order_moves(bd, moves, color, tt_move, ply):
        undo = make_move(bd, sr, sc, er, ec)
        score = -alphabeta(bd, next_turn, depth - 1, -beta, -alpha, ply + 1)
        unmake_move(bd, undo)

        if score > best:
            best = score
            best_move = (sr, sc, er, ec)

        if score >= beta:  # 🔥 KILLER + HISTORY UPDATE
            if killers_active and bd["grid"][sr][sc] != '.':
                if KILLERS[ply][0] != (sr, sc, er, ec):
                    KILLERS[ply][1] = KILLERS[ply][0]
                    KILLERS[ply][0] = (sr, sc, er, ec)
            if history_active:
                HISTORY[sq(sr, sc)][sq(er, ec)] += depth * depth
            return beta

        alpha = max(alpha, score)

        if alpha >= beta:
            # Update history
            HISTORY[sq(sr, sc)][sq(er, ec)] += depth * depth
            break

    # Store TT
    flag = EXACT
    if best <= orig_alpha:
        flag = UPPER
    elif best >= beta:
        flag = LOWER

    if not is_repetition() and not fifty_move_draw(bd):
        TT[h] = (best, depth, flag, best_move)
        pass

    return best
def quiescence(bd, turn, alpha, beta, qdepth=0):
    #return evaluate(bd,turn)
    if is_repetition() or fifty_move_draw(bd):
        return 0

    moves = all_moves_for_board(bd, turn)

    if in_check(bd, turn) and not moves:
        return -INF
    stand_pat = evaluate(bd, turn)

    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    if qdepth >= q_depth:
        return alpha

    moves = quiescence_moves(bd, turn, moves)
    moves.sort(key=lambda m: capture_score(bd, m), reverse=True)

    for move in moves:
        undo = make_move(bd, *move)

        # IMPORTANT: skip illegal captures
        if in_check(bd, turn):
            unmake_move(bd, undo)
            continue

        score = -quiescence(bd, opponent(turn), -beta, -alpha, qdepth + 1)
        unmake_move(bd, undo)

        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha

def order_moves(bd, moves, color, tt_move=None, ply=0):
    """
    Orders moves:
    1) Captures
    2) Checks
    3) Quiet moves
    """

    ordered = []

    if tt_move and tt_move in moves:
        ordered.append((10_000_000, tt_move))
        moves = [m for m in moves if m != tt_move]

    enemy = opponent(color)

    for move in moves:
        r1, c1, r2, c2 = move
        piece = bd["grid"][r1][c1]
        target = bd["grid"][r2][c2]

        score = 0

        # Capture
        if target != '.':
            score += 1000 + VALUES[target.lower()] - VALUES[piece.lower()] + capture_score(bd,move)

        # En passant capture
        if piece.lower() == 'p' and bd["en_passant"] == (r2, c2):
            score += 1000

        # Promotions
        if piece.lower() == 'p' and (r2==0 or r2==7):
            score += 10000


        # Check
        #undo = make_move(bd, r1, c1, r2, c2)
        #if in_check(bd, enemy):
        #    score += 500
        #unmake_move(bd, undo)

        #killer moves
        if move == KILLERS[ply][0]:

            score += 900
        elif move == KILLERS[ply][1]:
            score += 800

        score += HISTORY[sq(r1, c1)][sq(r2, c2)]

        ordered.append((score, move))

    ordered.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in ordered]
def capture_score(bd, move):
    r1, c1, r2, c2 = move
    attacker = bd["grid"][r1][c1].lower()
    target = bd["grid"][r2][c2]

    if target == '.':
        return 100  # en passant

    return VALUES[target.lower()] * 10 - VALUES[attacker]
def quiescence_moves(bd, turn, move_list):
    moves = []
    for r1, c1, r2, c2 in move_list:
        piece = bd["grid"][r1][c1]
        target = bd["grid"][r2][c2]
        if target != '.' or (
            piece.lower() == 'p' and bd["en_passant"] == (r2, c2)
        ):
            moves.append((r1, c1, r2, c2))

        undo = make_move(bd, r1, c1, r2, c2)
        if in_check(bd, opponent(turn)):
            moves.append((r1, c1, r2, c2))
        unmake_move(bd, undo)

    return moves

def board_to_fen(bd, turn):
   rows = []
   for r in range(8):
       empty = 0
       row = ""
       for c in range(8):
           p = bd["grid"][r][c]
           if p == '.':
               empty += 1
           else:
               if empty:
                   row += str(empty)
                   empty = 0
               row += p
       if empty:
           row += str(empty)
       rows.append(row)


   board_part = "/".join(rows)


   # side to move
   stm = "w" if turn == "white" else "b"


   # castling
   castling = ""
   if bd["castling"]["white_king"]: castling += "K"
   if bd["castling"]["white_queen"]: castling += "Q"
   if bd["castling"]["black_king"]: castling += "k"
   if bd["castling"]["black_queen"]: castling += "q"
   if not castling:
       castling = "-"


   # en passant
   if bd["en_passant"] is None:
       ep = "-"
   else:
       r, c = bd["en_passant"]
       ep = chr(ord('a') + c) + str(8 - r)


   return f"{board_part} {stm} {castling} {ep} 0 1"
def stockfish_move(bd, turn, time_ms=2):
   fen = board_to_fen(bd, turn)
   board = chess.Board(fen)

   result = engine.play(
       board,
       chess.engine.Limit(time=time_ms / 1000)
   )

   move = result.move
   if move is None:
       return None

   # convert UCI -> your coords
   sr = 8 - int(move.uci()[1])
   sc = ord(move.uci()[0]) - ord('a')
   er = 8 - int(move.uci()[3])
   ec = ord(move.uci()[2]) - ord('a')
   return (sr, sc, er, ec)
def stockfish_eval_white(info):
   score = info["score"].white()

   if score.is_mate():
       return 100000 if score.mate() > 0 else -100000

   return score.score()
#endregion

#region ---------------- MOVE EVALUATION ----------------
VALUES={'p':100,'n':300,'b':300,'r':500,'q':900,'k':0} #piece values
INF = 100000

ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
PAWN_ATTACK_PENALTY = {
    'n': 35,
    'b': 30,
    'r': 20,
    'q': 10,
}
KNIGHT_OUTPOST_BONUS = 40

MG_PST = {
    'p': [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [ 50, 50, 50, 50, 50, 50, 50, 50],
        [ 10, 10, 20, 30, 30, 20, 10, 10],
        [  5,  5, 10, 25, 25, 10,  5,  5],
        [  0,  0,  0, 20, 20,  0,  0,  0],
        [  5, -5,-10, 20, 20,-10, -5,  5],
        [  5, 10, 10,-20,-20, 10, 10,  5],
        [  0,  0,  0,  0,  0,  0,  0,  0]
    ],

    'n': [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ],

    'b': [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ],

    'r': [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  5, 10, 10, 10, 10, 10, 10,  5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [  0,  0,  0,  5,  5,  0,  0,  0]
    ],

    'q': [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ],

    'k': [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20]
    ]
}
EG_PST = {
'p': 2*[
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -10, -10,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
]
,

    'n': [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]
,

    'b': [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]
,

    'r': [
    [  0,   0,   0,   5,   5,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   5,  10,  10,   5,   0,   0],
]
,

    'q': [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
]
,

    'k': [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -10,   0,   0,   0,   0, -10, -30],
    [-30,   0,  20,  30,  30,  20,   0, -30],
    [-30,   0,  30,  40,  40,  30,   0, -30],
    [-30,   0,  30,  40,  40,  30,   0, -30],
    [-30,   0,  20,  30,  30,  20,   0, -30],
    [-30, -10,   0,   0,   0,   0, -10, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]

}
PHASE_WEIGHTS = {
    'p': 0,
    'n': 1,
    'b': 1,
    'r': 2,
    'q': 4,
    'k': 0,
}

def evaluate(bd, turn):
    global nodes
    nodes += 1
    base = bd["material"] + bd["pst"] + bd["passed"] + 10*(len(bd["pieces"][turn]["b"]) == 2)

    if endgame_phase(bd):
        base += endgame_eval(bd, turn)

    #base += eval_rooks_files(bd, "white")
    #base -= eval_rooks_files(bd, "black")

    #base += eval_pawn_attacks(bd, "white")
    #base -= eval_pawn_attacks(bd, "black")

    #base += eval_knight_outposts(bd, "white")
    #base -= eval_knight_outposts(bd, "black")

    base += eco_pawn_eval(bd, turn)

    return base if turn == "white" else -base

def pawn_eval(bd):
    h = bd["pawn_hash"]

    if h in PAWN_TT:
        return PAWN_TT[h]

    score = (
        pawn_structure_score(bd, "white")
        - pawn_structure_score(bd, "black")
    )

    PAWN_TT[h] = score
    return score
def pawn_structure_score(bd, color):
    pawns = bd["pieces"][color]['p']
    opp = opponent(color)
    opp_pawns = bd["pieces"][opp]['p']

    score = 0
    score += doubled_pawns(pawns, color)
    score += isolated_pawns(pawns, color)
    score += passed_pawns(pawns, opp_pawns, color)*(bd["phase"]/16+1)

    return score
def doubled_pawns(pawns, color):
    file_counts = [0]*8
    for r, c in pawns:
        file_counts[c] += 1

    penalty = 0
    for cnt in file_counts:
        if cnt > 1:
            penalty -= (cnt - 1) * 15

    return penalty
def isolated_pawns(pawns, color):
    pawn_files = {c for _, c in pawns}
    penalty = 0

    for r, c in pawns:
        if (c-1 not in pawn_files) and (c+1 not in pawn_files):
            penalty -= 20

    return penalty
def passed_pawns(pawns, opp_pawns, color):
    bonus = 0
    direction = -1 if color == "white" else 1

    opp_by_file = {}
    for r, c in opp_pawns:
        opp_by_file.setdefault(c, []).append(r)

    for r, c in pawns:
        blocked = False
        for fc in (c-1, c, c+1):
            if fc not in opp_by_file:
                continue
            for o_row in opp_by_file[fc]:
                if (o_row - r) * direction > 0:
                    blocked = True
                    break
            if blocked:
                break

        if not blocked:
            advance = (6 - r) if color == "white" else (r - 1)
            bonus += 20 + advance * 10

    return bonus

def eco_pawn_eval(bd, color):
    eco = bd.get("eco")
    if not eco:
        return 0

    if eco == 'A':
        return eco_pawn_eval_A(bd, color)
    if eco == 'B':
        return eco_pawn_eval_B(bd, color)
    if eco == 'C':
        return eco_pawn_eval_C(bd, color)
    if eco == 'D':
        return eco_pawn_eval_D(bd, color)
    if eco == 'E':
        return eco_pawn_eval_E(bd, color)

    return 0
def eco_pawn_eval_A(bd, color):
    w = bd["phase"]/8 - 1
    if w <= 0:
        return 0

    pawns = bd["pieces"][color]['p']
    score = 0

    # Reward c-pawn advance
    if (5 if color == "white" else 2, 2) in pawns:  # c3/c6
        score += 8
    if (4 if color == "white" else 3, 2) in pawns:  # c4/c5
        score += 12
    if board["opening_name"] == "Stonewall Attack" and color == "white":
        core = [(5, 2), (4, 3), (5, 4), (4, 5)]  # c3 d4 e3 f4 stonewall setup

        count = sum(1 for sq in core if sq in pawns)

        if count == 3:
            score += 5
        elif count == 4:
            score += 15
        else:
            score-=5


    # Central dark-square clamp
    if (4, 5) in pawns and (3, 4) not in pawns:
        score += 5

    return int(score * w)
def eco_pawn_eval_B(bd, color):
    w = bd["phase"]/8 - 1
    if w <= 0:
        return 0

    pawns = bd["pieces"][color]['p']
    score = 0

    if color == "white":
        if (3, 3) in pawns and (2, 4) in pawns:
            score += 5  # French-style solidity
        else:
            score -= 5
        if (3, 4) in bd["pieces"][color]['p']:  # advanced e5 pawn
            score += 3
        else:
            score -= 3

    # c-pawn tension
    if (5 if color == "white" else 2, 2) in pawns:
        score += 10

    return int(score * w)
def eco_pawn_eval_C(bd, color):
    w = bd["phase"]/8 - 1
    if w <= 0:
        return 0

    pawns = bd["pieces"][color]['p']
    score = 0

    # Strong central pawn presence
    if (4 if color == "white" else 3, 4) in pawns:  # e-pawn
        score += 15
    if (4 if color == "white" else 3, 3) in pawns:  # d-pawn
        score += 10

    if color == "white":
        if (3,1) in bd["pieces"]["white"]['b']:
            score += 5
        else:
            score -= 5

    # c4 still useful but less important
    if (5 if color == "white" else 2, 2) in pawns:
        score += 5

    return int(score * w)
def eco_pawn_eval_D(bd, color):
    w = bd["phase"]/8 - 1
    if w <= 0:
        return 0

    pawns = bd["pieces"][color]['p']
    score = 0

    # d-pawn priority
    if (4 if color == "white" else 3, 3) in pawns:
        score += 12

    # c-pawn supports d4
    if (5 if color == "white" else 2, 2) in pawns:
        score += 8
    if color == "white":
        if (4, 2) in bd["pieces"]["white"]['p']:  # pawn on c4
            score += 6
        else:
            score -= 6
        if ((5, 4) in bd["pieces"]["white"]['p'] or (5, 4) in bd["pieces"]["white"]['b']) and color == "white":
            score += 3
        else:
            score -= 3

    if (4, 3) in bd["pieces"]["white"]['p'] and (3, 3) in bd["pieces"]["black"]['p']:
        score += 4  # symmetrical exchange structure

    return int(score * w)
def eco_pawn_eval_E(bd, color):
    w = bd["phase"]/8 - 1
    if w <= 0:
        return 0

    pawns = bd["pieces"][color]['p']
    score = 0

    # Central restraint
    if (5 if color == "white" else 2, 3) in pawns:
        score += 6

    # f-pawn often signals kingside play
    if (5 if color == "white" else 2, 5) in pawns:
        score += 8

    return int(score * w)

def endgame_phase(bd):
    # centipawns
    phase = abs(bd["phase"])

    # threshold ≈ R + minor
    return phase <= 4
def king_distance(bd):
    wk = bd["king_pos"]["white"]
    bk = bd["king_pos"]["black"]
    return abs(wk[0] - bk[0]) + abs(wk[1] - bk[1])
def king_edge_distance(r, c):
    return min(r, 7 - r) + min(c, 7 - c)
def endgame_eval(bd, turn):
    us = turn
    them = opponent(turn)

    ur, uc = bd["king_pos"][us]
    tr, tc = bd["king_pos"][them]

    score = 0

    # Encourage king activity
    score -= king_distance(bd) * 5

    # Push enemy king to edge
    score += (7 - king_edge_distance(tr, tc)) * 10

    return score

def eval_rooks_files(bd, color):
    bonus = 0
    opp = opponent(color)

    pawns_white = {c for r, c in bd["pieces"]["white"]["p"]}
    pawns_black = {c for r, c in bd["pieces"]["black"]["p"]}

    for r, c in bd["pieces"][color]["r"]:
        if color == "white":
            has_friendly = c in pawns_white
            has_enemy = c in pawns_black
        else:
            has_friendly = c in pawns_black
            has_enemy = c in pawns_white

        if not has_friendly and not has_enemy:
            bonus += ROOK_OPEN_FILE_BONUS
        elif not has_friendly and has_enemy:
            bonus += ROOK_SEMI_OPEN_FILE_BONUS

    return bonus
def eval_pawn_attacks(bd, color):
    penalty = 0
    opp = opponent(color)

    for pr, pc in bd["pieces"][opp]['p']:
        direction = 1 if opp == "white" else -1
        for dc in (-1, 1):
            tr, tc = pr + direction, pc + dc
            if 0 <= tr < 8 and 0 <= tc < 8:
                target = bd["grid"][tr][tc]
                if target != '.' and (
                    (color == "white" and target.isupper()) or
                    (color == "black" and target.islower())
                ):
                    pt = target.lower()
                    penalty += PAWN_ATTACK_PENALTY.get(pt, 0)

    return -penalty
def eval_knight_outposts(bd, color):
    bonus = 0
    opp = opponent(color)

    pawn_dir = 1 if color == "white" else -1
    enemy_pawn_dir = -pawn_dir

    for r, c in bd["pieces"][color]['n']:
        # Must be in enemy territory
        if (color == "white" and r > 3) or (color == "black" and r < 4):
            continue

        # Supported by pawn
        supported = False
        for dc in (-1, 1):
            sr = r - pawn_dir
            sc = c + dc
            if (sr, sc) in bd["pieces"][color]['p']:
                supported = True
                break
        if not supported:
            continue

        # Not attackable by enemy pawn
        attacked = False
        for dc in (-1, 1):
            ar = r - enemy_pawn_dir
            ac = c + dc
            if (ar, ac) in bd["pieces"][opp]['p']:
                attacked = True
                break

        if not attacked:
            bonus += KNIGHT_OUTPOST_BONUS

    return bonus

def pst_value(pt, r, c, color, bd):
    if color == "white":
        mg = MG_PST[pt][r][c]
        eg = EG_PST[pt][r][c]
    else:
        mg = -MG_PST[pt][7 - r][c]
        eg = -EG_PST[pt][7 - r][c]

    phase = bd["phase"]
    eg_w = (16 - phase) / 16
    mg_w = 1.0 - eg_w

    return mg * mg_w + eg * eg_w

def is_repetition():
    if not POSITION_HISTORY:
        return False

    current = POSITION_HISTORY[-1]
    count = 0

    # Only need to check back until last irreversible move
    for h in reversed(POSITION_HISTORY[:-1]):
        if h == current:
            count += 1
            if count >= 2:  # current position makes it the 3rd
                return True
    return False
def fifty_move_draw(bd):
    return bd["halfmove_clock"] >= 100
#endregion

# region ---------------- GAME STATE ----------------
random_nums = [10333405229380957596, 2845955593063092058, 5555654593594871304, 3344389014250026926,
               3122328447014561036, 15310476613626113862, 9610972471107806522, 12100999972405130138,
               8135837782334848880, 8864233081127514773, 3931119832596259481, 5948667151737291575, 8271195169857813383,
               15709563716407736097, 15364941764661514516, 10657178980788831838, 18319651717548027621,
               2898787316831901566, 10176060493730393222, 1960464341236732732, 17263936515512432911,
               4810410454346779567, 425861619111063, 3464858955199058563, 2958785963048336006, 17180439751915386712,
               10878543363932969677, 4943394655243782526, 15071540435686099527, 4337187280877572483,
               11992880388372081644, 9174611086670129513, 6712532852438397729, 15884060040952686688,
               2021917893054190834, 10738743202903633069, 8149856126668335879, 16144986158873084679,
               16159029570282286801, 5443972132529251959, 16310780328355849218, 17853564445256637968,
               8949936093581076198, 15576973020744906388, 8170233719244137382, 5498286255991983855, 1220515634625681384,
               12354447707845694135, 9743366923190463876, 17929771845861387651, 7292875059340752540,
               4577454490594580000, 3257754793226943749, 15522049912520698849, 1949888369054325990, 5815880887465891629,
               1290423990089744283, 3794679449968240857, 12661942184484736218, 11505157190382552778,
               13957141432844239859, 6257986415296806443, 13969052440017719928, 10374723717006043271,
               12472454499367654117, 15396056574721895908, 3806999433765445074, 5119299321164021355,
               9404694795182981561, 9361103398939879780, 12079341791803685969, 967795244508692043, 13161128134197059351,
               542277789116479525, 1856484581703228826, 8126310165989427459, 10599183971148376605, 15741478458763564961,
               17307050533211593913, 9627138911443208808, 16828901891279117641, 14773071064896585224,
               15748197783020352264, 8545380957555492651, 7254503587555449909, 16331626830373391862,
               14798007346546631073, 8812675538186624813, 7738564939505065204, 1400033256067458762,
               16645224920977780879, 13222133040698529083, 15547609482794038295, 3041024829037250659,
               3465108710436768441, 12844226247586224161, 14497597992029566508, 13352221998787378998,
               4869835072615098499, 9420584704785674691, 5221466349787445197, 18129197659796256915, 1799938404344280044,
               7292849973438511816, 4583209458514154602, 15012630393974908711, 11051479372508600267,
               13798485318908887225, 9116379073736082902, 2292844428448342091, 16792072819360499131,
               10314055198202365376, 4779323059863146953, 11569881563381024632, 10883876838806891937,
               6696235222639917976, 5453139597263950637, 15612030376108878036, 5116070868288012338, 9939155182705129134,
               15457774586197629202, 335038915345056939, 319616843419063367, 15594962081776746455, 11463871160076043691,
               134205428722231655, 6315959864315611235, 6213792421354884331, 8120919171644498908, 8868632822570644284,
               210001292931445984, 17136876951592681778, 4279735878496317459, 3757466962515965228, 761822430316296588,
               12689221986685692245, 9490010134614485663, 1910070675751090337, 4621773589720601284, 8874885302745577394,
               18068053824955704235, 8566058660390355813, 9093552232826280897, 8884035096536711878, 1546355525808348243,
               15124887822099098067, 6563080063301772598, 2486699335416417550, 19303525121538065, 8870208367854004949,
               13413542733658238801, 11625746299326573294, 3414378168052937230, 6613267156271595745,
               5745278208663794559, 16413988530624789362, 3246141890112876946, 7554181998717568454,
               10255331515808177833, 8539973993849966497, 102069553557967693217011713439460312231, 479347332660210610,
               11518725046790933315, 7972156270936970802, 15204720696545765390, 16198501782893359384,
               1154711741963595329, 132983499281017165, 4540146311083168913, 11500725254541864666, 8105409497096457332,
               2096903634388762728, 14467226980936777355, 9125248561165186132, 10887072984451517056,
               10227101305771472359, 3044508608785695024, 8340962780683547461, 13759609875266877923,
               9090091251253354248, 1685970101385925044, 2097765445609344297, 2532083319479202997, 11182487744945489881,
               15061491421236069898, 8714432361452740095, 4419442509603367629, 5028268250952899151, 345243517584360,
               17501186510744747252, 18256197953582361867, 869736101123260367, 17628890832073833865,
               2774718494948837992, 7961255522350048363, 17574973417328974018, 14707388304036814389,
               16205595217046366642, 4449978607449260587, 6402843305730634370, 13563173847002992420,
               9813347926675894271, 9090507561252241153, 4942000013654224773, 16858167536358920068,
               12576743789548831630, 7881324247215346870, 11332266382244448250, 11774525976613004275,
               15102215981484080159, 11508072310035869866, 1587607397111747273, 6048991191504673775,
               14224940991857039077, 17978680498760286254, 14832032761625772915, 17489241679554648657,
               11170594066999756862, 17607032702678915184, 15668885190473092332, 208848654057604363,
               14985497667052589775, 2656325571535609183, 6269366259863678734, 11099037149017661586,
               15789687237569140193, 13538453187020824899, 12431443171648576455, 18357767858425791388,
               1299743164555271113, 10119743959729654182, 12542885796207224966, 3157835018163503136,
               9420553790516387974, 9880692943064840334, 616850879683837894, 1626439506631363629, 16552620019817847746,
               6021330455333714428, 9122675459355348545, 12620172581514576684, 15452085262637871397,
               3776492740725288986, 4761142956017132147, 14585634911771014161, 2864170541564634751,
               17680950492892163209, 17014434729195712245, 10480818977448710603, 12239358099757883900,
               8509869149401794875, 4784575312957937239, 5955705017734837207, 18244386775886843875, 8592788065377189409,
               16404816893075546240, 4559155336079668999, 16980652206037746090, 16289025099231484366,
               10658351340252255186, 12632509176019471006, 9700541193387880168, 16826402361086759361,
               16379849961731905538, 5726254838033985306, 5065925675705926145, 6580102410514059999, 5509603224880173629,
               10129194225568216321, 11539453422454223742, 12795251640615501849, 9696618095936484415,
               14506955870751818947, 5619213736829696796, 5206059727279193739, 13768113839842024065,
               9463217674542620863, 215658778309208377, 5303613378395297980, 6656929944442386301, 4463726871381016811,
               13743002578785473348, 15669169672863032490, 4454644597085378200, 13882754280743126747,
               11597631920271990366, 2385856239551219321, 5049235190142056741, 5815440131657789820, 3238458444227873078,
               1307593501528688696, 16636076260774200292, 14485601902987402804, 11722364491183088213,
               9240304382745302567, 3716267729676875510, 6300202574864480667, 6309089639439885941, 4567847374131786566,
               5141617922971922871, 13724373383655974786, 11965986385120397418, 2883290124738138834,
               4619038923248318131, 11272611976164368811, 17247238098806271600, 11526130365499440252,
               12350411483598273794, 5964827568055721534, 6083677990279023210, 15290531256936441334,
               1392185335699687581, 1337979267759328599, 17801613634240442458, 4672544550285858362,
               15826777105033047794, 439521105379679511, 14337720886593063485, 519825808424035301, 7258628903111207806,
               10927783742922400061, 7893242041188940839, 1050189761602689953, 16701994763056368544,
               12561053908912981279, 2518870342361011004, 6675037227001384550, 2548554239851006234, 2252690395001638106,
               9324367467532813654, 2024701245265502937, 166174727139723445, 6074633565733664297, 1859862474236470138,
               16858413033220409306, 11000242238444891414, 1053504724449944053, 17374653385596035164,
               3876463564140588009, 14014398360807191048, 10551959233433854627, 432228681435796134,
               11701515436507904550, 6949422208655383251, 5277590929197037698, 9561404066796575339,
               10743779221045601478, 6718608850268642640, 17713145785151698595, 9231375358621164190,
               2243875736939644926, 9958654298718960161, 13081205206978248072, 3245498401079226850,
               15088501948376521925, 16957369048382618573, 14407340543766185768, 652375221028024140,
               13338670110852396924, 2118999380223384801, 8488645387551935803, 4331355963827606248, 2479785917224283260,
               13644146973834440903, 3118551649634799951, 13070688733980315888, 13679698271734671424,
               15240289418296315646, 1139363903492322452, 15472427117223757359, 5640044149928144283,
               11882637659579481610, 5895388232927832804, 1161134879417526933, 16657043279186642842,
               10373605971651213952, 15134105255291274088, 3993340980218751080, 3114630334806680447,
               18060540997754049576, 7579233495649087384, 12071612272108620319, 9408518411836511951,
               12539847131214501949, 12348150839266177242, 17566150522571138675, 6084146188547008349,
               13454951581341660322, 6700672018636314709, 12888008826803432085, 9108704819126429094,
               6950435963242266290, 2958830906731836721, 8743791538366470749, 5848553911880588006, 10872004130150235116,
               8396861743596247885, 13731652842246385525, 15154501454985001618, 18043468484339272011,
               3591711710718771603, 14013913134733541746, 11614526876128259340, 9577129416009737251,
               10813293790800158824, 9603680548322118917, 7058061554912496838, 17997014351186603091,
               10600767428481001266, 16655151292571869602, 10465471959796463938, 7102130288365927186,
               17706094352439408131, 126301117089195138, 6398340010460024522, 3249953911177750428, 5962429815796348315,
               1372754807110047870, 16941254468436004193, 3102799185375388608, 5535903693188832155,
               13229795671839880570, 10900190483042355223, 10296717423857970014, 8053950612936417323,
               15815256902377882673, 16098940258513625353, 91400898369250757, 16573235207823950220, 7988214401740616231,
               614897830007309125, 4496466465442296810, 7233610375993073419, 4626680905871627374, 16649418983741984881,
               7861653819462393088, 7217002638387760659, 7587312518311943606, 13724308311425712054, 3342642923328174775,
               11765525801414785873, 3982312221328254700, 4331513078004280058, 10413339208062755923,
               4545752569123147332, 1309268436412613159, 14751899811732110287, 17890868519233939942,
               2951976582325348472, 6103019931331915626, 13018624862119463117, 8480168829066249345, 6553159255542282694,
               12010435939006023357, 14289530356844154063, 3075325822185967996, 14986504491438552784,
               11715811921291079601, 10797870829973568415, 13296451726880093087, 2292758561780602633,
               10020376459440699580, 686733498226660922, 12031149337254684119, 16252648677737034816,
               14020274765878059778, 10103131026683331214, 8072950802965447176, 12207780996041738780,
               13621022539169246381, 17470091280557136537, 11467657560378563474, 6116184823607645057,
               5998965076257848459, 47488578411891131, 6607240841620269417, 13217650300290654370, 6190015603427810103,
               13970326282648110427, 374811800297982897, 12016517524005662304, 2186123927589182405, 8919122682320774490,
               7063392563016657032, 14947495723771989422, 1342806689905567092, 5161687297272764534, 1036534437628364764,
               9808405721383719697, 12005971095320497567, 5011312736401923295, 8411180458932627347,
               14979890799584818068, 9695235695470572760, 1314345139325355474, 4077142408073465661, 1687409842389548795,
               14535491554048914092, 12564026286757496718, 12040357640985859536, 1722096911597907185,
               7677479002160247088, 18119848801010636871, 17805506329259228846, 15143540562689838874,
               7407401916073186756, 347554027655266751, 13962201046825958147, 17052488209214965166,
               17158463281880312503, 18116991130805065485, 4964805959238956669, 3407721655619936020, 877418074089585716,
               16179873663944157705, 15073952446910163284, 13873624997727109690, 6202500148498741394,
               11873104576788809072, 9067234028108601304, 15365916810984266050, 9824083192736146870,
               6546004925589912052, 8752132714863247119, 16221183392281834910, 17060638500491764577,
               1981962973792427508, 7865031364331068979, 8712266576764974860, 18344812043654675498, 3991263906181083837,
               6290788480757421759, 14284695782133942469, 2292003977962689028, 10512236002245393733,
               9099153591199347951, 13639777264769249562, 6136807030425945470, 16350321620864457877,
               4277474565271741616, 14801427344085573423, 16284514657991069695, 10251236467292546316,
               5378614133092716598, 7198032582753385214, 4884507684639232302, 10858577945503197611, 851891517333492961,
               20725325125384313, 2123544383803692749, 3107584211126583283, 16795646889751016983, 3495589542100667354,
               13704526649552736666, 14462514491258018299, 11186467144953917000, 6595282277927965232,
               4232118338470176077, 8562505461920114606, 4937744948535537695, 18124570573946921140,
               18169484010194656926, 1809263646873319416, 8333561781499082201, 15335303234344312394,
               3360230493804493117, 4894673542181709726, 12981701706260122278, 11249101239740975633,
               16852057857256248898, 8702895657814645191, 5268393043799833915, 14117064889335292283,
               8881516933192703570, 12768256619523031565, 6359496236223616615, 7393424634625242646,
               13351637528683123228, 11867217396375460836, 1024184204059314535, 12841665749767818437,
               16892336369423635345, 13514981648016821182, 2432750109948887001, 2782865748986633014,
               8102328225010077477, 1290130327082460532, 5703180609187359267, 858974748432959727, 476593418895050000,
               12464608924969360085, 7314106122870194397, 6516292085269488896, 9069177514976645646, 5235127973728418566,
               15714307846379624758, 2651753811034952748, 8074258336693844345, 5564210978063733676,
               11499836101991457778, 12796174023000063552, 2463909423016555221, 12596677740591173412,
               10835426591930562242, 1462622494568909548, 17159906000923175390, 6490537668287666042,
               12024110141757942368, 16822169426037276068, 17073259779709727504, 16880439620767873053,
               4668622575625594504, 9473597058790174688, 13779660119239042324, 5201081756838048432, 3736931774724611643,
               9906296474943822752, 13779967920861470173, 10274181708940553639, 1048549740179126807,
               6799848914885357291, 11310093976128702140, 5974707930368231300, 11068035473565837340,
               3640787048452028865, 1145722209139464905, 4412474567823738542, 18046942662908024712, 2380642817319861986,
               16969987044766487668, 7347409451537330949, 13896891667480573388, 89967562992636179, 3477023944398842609,
               13086363993165569596, 7105109951338624838, 11904291980454388909, 16971588055877233974,
               4666497201661587559, 16840853923536607019, 2410185210100118301, 1840442706892270560,
               16460252038269329057, 6032545258028702434, 3501026143058489378, 8561545229113963857, 8796019860498841063,
               9734019357456631653, 6357140608091689290, 4130877390430424067, 15642282171222063682, 410424905472370072,
               2289574082241124143, 14380220655313725217, 3428028340881399233, 17498604737059439546,
               6673315607440095915, 7908308455706499580, 11526813710522246654, 1794034022948675943,
               14131779170260615221, 2776169411595490508, 10461640415040133757, 1646218523293435521,
               7678355896647020121, 8686592648660839046, 5915616540241608005, 18402937882554770424, 1033369596429489426,
               13124710305306371644, 12432462761084557386, 13229205381180367512, 17402188241511216876,
               14084429893397540588, 9735976568537314578, 9054864775086400548, 4369316902200518396,
               10222784413823500425, 12802552278316022544, 14712310424281419996, 7574829401297618355,
               5822573341889966088, 12381659858172323603, 10405423887687597686, 16212457304645263605,
               9070113799537588339, 2567909795961355713, 11139785702641801558, 10900036279776995016,
               7220491663789549912, 15782233371931666046, 10574436947684975196, 2079939645301681119,
               14925623644685284404, 15006748577471509888, 11361572961102338800, 17580691958370089296,
               16645971733593835965, 6049371592592243883, 2859953135754940287, 14726863064201207791,
               14990904927425544183, 10556180628685624324, 11529085032432707228, 10392024490805461626,
               5881590564991810471, 3564014853478565377, 2606081643001930028, 1087913762068759531, 8369312996902516726,
               11392540644598799165, 12939069450600240861, 11498686984401063064, 2509895286910852754,
               12493139651374041272, 14766473254209498663, 2579093280474853426, 3575116879632855204,
               9329491043563366901, 3009963496136604408, 5865445681515216083, 5719154039786343949, 4074810444812748069,
               13177135512471797513, 10640796810483737534, 11199997338201735824, 13006093904390291227,
               9938498104796389007, 12845671285957958917, 12919927865967978441, 13487724589774025401,
               2783187049795317795, 1355462942726209464, 9124830463541440437, 9684973046520717184, 876887303859278640,
               5990306611375191334, 1557798007399528335, 12006867803631039364, 6216183784938534780,
               12533672880398513669, 13294538252512925781, 16826449414833563186, 13447084884097474589,
               10161592024264266245, 18343156354685255633, 10916468183279113291, 6927296533127002538,
               16563249564280023235, 9270003909352243522, 10475341367624674272, 10405554461090103700,
               2619233219982985388, 8795896012646879172, 16352621696118967874, 12557679678197098232,
               7825734219598888254, 5849597394655077980, 4463132285556757009, 18087838601344324037, 4092221544171581259,
               3880506546522396469, 16651018823924988243, 6377226142499299319, 1476393091936652273, 5766162687152803940,
               9849439598385445775, 3068463824390333040, 17734602789602182665, 17637113033445019187,
               12254386308806423990, 5613643126627224949, 12043843731265863363, 9132027922904460452,
               2818847279399441120, 4919227834647774834, 2520673437519261468, 17731889365355996545,
               14564380366809947556, 16644281742949681065, 14315782704578896218, 4450491535300015951,
               1241337160839839954, 4642031461954021193, 2678856591773170962, 12468212425329086392, 4181051466412090676,
               15354243443690553529, 14659362075667870661, 2775982295746971228, 15402573491644137290,
               15877100283889681000, 18421729528573895920, 7169389014352349882]  # list of random 64-bit numbers

POSITION_HISTORY = []
Z_PIECES = {}
Z_TURN = random_nums.pop()
Z_CASTLE = {}
Z_EP = {}

PIECES = "PNBRQKpnbrqk"
for r in range(8):
    for c in range(8):
        for p in PIECES:
            Z_PIECES[(r, c, p)] = random_nums.pop()
for k in ["white_king", "white_queen", "black_king", "black_queen"]:
    Z_CASTLE[k] = random_nums.pop()
for f in range(8):
    Z_EP[f] = random_nums.pop()

TT = {}
PAWN_TT = {}
EXACT = 0
LOWER = 1
UPPER = 2

KILLERS = [[None, None] for _ in range(64)]
HISTORY = [[0] * 64 for _ in range(64)]

FILES = "abcdefgh"
RANKS = "87654321"

def sq_to_alg(r, c):
    return FILES[c] + RANKS[r]

class Undo:
    def __init__(self, sr, sc, er, ec, captured, ep, castling, promoted):
        self.sr = sr
        self.sc = sc
        self.er = er
        self.ec = ec
        self.captured = captured
        self.ep = ep
        self.castling = castling
        self.promoted = promoted

        # special-move flags
        self.ep_capture = False
        self.ep_pawn_pos = None
        self.castle = False
        self.rook_from = None
        self.rook_to = None

        # incremental snapshots (set in make_move)
        self.material = None
        self.pst = None
        self.zobrist = None
        self.king_pos = None
        self.phase = None
        self.pawn_hash = None
        self.halfmove_clock = None
        self.opening_name = None
        self.eco = None

def init_incremental_state(bd):
    bd["pieces"] = {
        "white": {p: set() for p in "pnbrqk"},
        "black": {p: set() for p in "pnbrqk"},
    }

    material = 0
    pst = 0
    g = bd["grid"]
    bd["phase"] = 24
    bd["pawn_hash"] = 0

    for r in range(8):
        for c in range(8):
            p = g[r][c]
            if p == '.':
                continue

            color = "white" if p.isupper() else "black"
            pt = p.lower()

            bd["pieces"][color][pt].add((r, c))

            pst += pst_value(pt, r, c, color, bd)

            if p == 'K':
                bd["king_pos"]["white"] = (r, c)
            elif p == 'k':
                bd["king_pos"]["black"] = (r, c)

            if p.lower() == 'p':
                bd["pawn_hash"] ^= Z_PIECES[(r, c, p)]

    bd["material"] = material
    bd["pst"] = pst
    bd["move_history"] = []  # list of SAN-like strings
    bd["ply"] = 0  # half-move counter
def new_board():
    bd={
        "grid": [
            list("rnbqkbnr"),
            list("pppppppp"),
            list("........"),
            list("........"),
            list("........"),
            list("........"),
            list("PPPPPPPP"),
            list("RNBQKBNR"),
        ],
        "en_passant": None,
        "castling": {
            "white_king": True,
            "white_queen": True,
            "black_king": True,
            "black_queen": True
        },
        # NEW
        "pieces": {
            "white": {p: set() for p in "pnbrqk"},
            "black": {p: set() for p in "pnbrqk"}
        },
        "king_pos": {
            "white": None,
            "black": None
        },
        "material": 0,  # evaluation base
        "zobrist": 0,
        "passed": 0,
        "halfmove_clock": 0,
        "opening_name": None,
        "eco": None

    }
    init_incremental_state(bd)
    return(bd)
def reset_game():
    global board, turn, POSITION_HISTORY, HISTORY, nodes, game_num, stockfish_eval, my_eval, time_log
    #record_state()
    stats.finalize()
    rows = stats.to_excel_rows()
    print_game_stats_csv(rows)
    #with open("results.csv", "a") as f:
        #f.write(rows)
        #f.write("\n")
    game_num += 1
    board = new_board()
    turn = "white"
    POSITION_HISTORY = []
    stockfish_eval = []
    my_eval = []
    nodes = 0
    time_log = []
    #turn = None

def stock_move_history(bd):
    moves = bd["move_history"]
    out = []

    for i in range(0, len(moves), 2):
        move_no = i // 2 + 1
        if i + 1 < len(moves):
            out.append(f"{move_no}. {moves[i]} {moves[i+1]}")
        else:
            out.append(f"{move_no}. {moves[i]}")
    return " ".join(out)
def move_to_san(bd, sr, sc, er, ec, promotion=None):
    piece = bd["grid"][sr][sc]
    capture = bd["grid"][er][ec] != '.'

    # --- CASTLING ---
    if piece.lower() == 'k':
        cs = castle_san(sr, sc, er, ec)
        if cs:
            return cs

    # --- PIECE LETTER ---
    if piece.lower() == 'p':
        prefix = ""
    else:
        prefix = piece.upper()

    # --- DISAMBIGUATION ---
    if piece.lower() != 'p':
        prefix += disambiguation(bd, piece, sr, sc, er, ec)

    # --- PAWN CAPTURE FILE ---
    if piece.lower() == 'p' and capture:
        prefix = FILES[sc]

    san = prefix
    if capture:
        san += "x"
    san += sq_to_alg(er, ec)

    # --- PROMOTION ---
    if promotion:
        san += "=" + promotion.upper()

    return san
def record_move(bd, sr, sc, er, ec, promotion=None):
    san = move_to_san(bd, sr, sc, er, ec, promotion)
    bd["move_history"].append(san)
    bd["ply"] += 1
def disambiguation(bd, piece, sr, sc, er, ec):
    others = same_piece_can_reach(bd, piece, sr, sc, er, ec)
    if not others:
        return ""

    same_file = any(c == sc for _, c in others)
    same_rank = any(r == sr for r, _ in others)

    if not same_file:
        return FILES[sc]
    if not same_rank:
        return RANKS[sr]
    return FILES[sc] + RANKS[sr]
def same_piece_can_reach(bd, piece, sr, sc, er, ec):
    color = "white" if piece.isupper() else "black"
    pt = piece.lower()

    others = []
    for r, c in bd["pieces"][color][pt]:
        moves = all_moves_for_board(bd, turn)
        if (r, c) == (sr, sc):
            continue
        if (r, c, er, ec) in moves:
            others.append((r, c))

    return others
def castle_san(sr, sc, er, ec):
    if abs(ec - sc) == 2:
        return "O-O" if ec == 6 else "O-O-O"
    return None

def rc_to_square(r, c):
    file = chr(ord('a') + c)
    rank = str(8 - r)
    return file + rank
def move_to_uci(bd, move):
    sr, sc, er, ec = move

    from_sq = rc_to_square(sr, sc)
    to_sq = rc_to_square(er, ec)

    piece = bd["grid"][sr][sc]

    uci = from_sq + to_sq

    # Promotion
    if piece.lower() == 'p' and (er == 0 or er == 7):
        # default to queen (recommended)
        uci += 'q'
    return uci
def print_game_stats_csv(stats_list):
    # stats_list = list of dicts (white + black entries)
    keys = list(stats_list[0].keys())
    print("\t".join(keys))
    for stats in stats_list:
        print("\t".join(str(stats[k]) for k in keys))




# endregion

if __name__ == "__main__":
    # ---------------- MAIN LOOP ----------------
    blunders = 0
    board = new_board()
    turn = "white"
    selected = None
    legal_moves = []
    clock = pygame.time.Clock()
    if visuals_enabled:
        draw()
    playing = True
    n = 0
    win = []
    start = pygame.time.get_ticks()
    nodes = 0

    while playing:
        clock.tick(60)
        move = get_move_for_turn(turn)

        if move:
            sr, sc, er, ec = move
            moves = all_moves_for_board(board,turn)
            record_move(board, sr, sc, er, ec, ('Q' if (er==0 or er==7) and (board["grid"][sr][sc]).lower()=='p' else None))
            uci = move_to_uci(board, move)
            sf_board = chess.Board(board_to_fen(board, turn))
            sf_board.push_uci(uci)
            new_evl = stockfish_eval_white(
                engine.analyse(sf_board, chess.engine.Limit(depth=10))
            )
            make_move(board, *move)
            fen = board_to_fen(board, turn)
            try:
                test_board = chess.Board(fen)
            except ValueError as e:
                print("BAD FEN:", fen)
                raise e
            #pygame.time.delay(100)

            stats.record_move(
                color=turn,
                eval_before=prev_evl,
                eval_after=new_evl,
                forced=(len(moves)==1)
            )
            prev_evl = new_evl
            turn = opponent(turn)

            if is_repetition():
                n+=1
                #print("Draw by repetition")
                win.append("repetition")
                reset_game()
                if n < 3:
                    start = pygame.time.get_ticks()

            if fifty_move_draw(board):
                n+=1
                #print("fifty_move_draw")
                win.append("fifty move draw")
                reset_game()
                if n < 3:
                    start = pygame.time.get_ticks()('')

        elif (turn != "white" or WHITE_PLAYER != HUMAN) and (turn != "black" or BLACK_PLAYER != HUMAN):
            n+=1
            #print(f"game evaluated {nodes} nodes in {(pygame.time.get_ticks() - start) / 1000} seconds with {blunders} blunders")
            if in_check(board, turn):
                #print(f"Checkmate! {'White' if turn == 'black' else 'Black'} wins.")
                win.append('White' if turn == 'black' else 'Black')
            elif is_repetition():
                #print("Draw by repetition")
                win.append("repetition")
            else:
                #print("Stalemate.")
                win.append("stalemate")
            reset_game()
            if n<3:
                start = pygame.time.get_ticks()
            else:
                #print(export_games_to_excel(games))
                playing = False
                #print(games)
                #print(f"white won {win.count('White')} out of {n} games, with {win.count('repetition')} repetitions and {win.count('stalemate')} stalemates")
        elif visuals_enabled:
            for e in pygame.event.get():
                draw()
                if e.type==pygame.QUIT:
                   pygame.quit(); sys.exit()
                if e.type==pygame.MOUSEBUTTONDOWN:
                   r,c=pygame.mouse.get_pos()[1]//SQ,pygame.mouse.get_pos()[0]//SQ
                   if selected:
                       if (r,c) in legal_moves:
                           make_move(board,*selected,r,c)
                           turn=opponent(turn)
                       selected=None; legal_moves=[]
                   else:
                       p=board["grid"][r][c]
                       if p!='.' and p.isupper()==(turn=="white"):
                           selected=(r,c)
                           legal_moves=[m[2:] for m in all_moves_for_board(board,turn) if m[:2]==(r,c)]
        if visuals_enabled:
            draw()
