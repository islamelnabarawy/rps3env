import logging
import random
import sys

from rps3env.opponents import BaseOpponent

# initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(tabs)s%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class MinMaxOpponent(BaseOpponent):
    def get_board_layout(self):
        layout = ['R', 'P', 'S'] * 3
        random.shuffle(layout)
        return layout

    def get_player_hand(self):
        hand = random.choice(['R', 'P', 'S'])
        return hand

    def get_next_move(self):
        move = None
        if self.iterative_deepening:
            for depth in range(1, self.depth_limit + 1):
                move = self.get_next_val(self._state, depth, True, root=True)
                # logger.debug('\n' + '=' * 50 + ' End Iteration %s ' + '=' * 50 + '\n', depth, extra={'tabs': ''})
        else:
            move = self.get_next_val(self._state, self.depth_limit, True, root=True)
        if move is None:
            return None
        result = self.get_move_code(move)
        return result

    @staticmethod
    def get_move_code(move):
        return '%s%s:%s%s' % (move[0] + move[1])

    def print_history_table(self):
        for state_hash, moves in self.history_table.items():
            print(state_hash)
            for m, s in moves.iteritems():
                print('\t', m, ':', s)

    def get_history_table_as_string(self):
        output = ''
        for state_hash, moves in self.history_table.items():
            output += state_hash + '\n\t'
            for m, s in moves.iteritems():
                output += self.get_move_code(m) + '=' + str(s) + ' '
            output += '\n'
        return output

    def __init__(self, depth_limit=4, heuristic_weights=(3, 1, -3), iterative=True):
        super(MinMaxOpponent, self).__init__()
        self.depth_limit = depth_limit
        self.history_table = {}
        self.heuristic_weights = heuristic_weights
        self.iterative_deepening = iterative

    def get_state_heuristic(self, state):
        captured = sum(state.captures)
        uncovered = 9 - state.counts[-1]
        lost = 9 - sum(state.player_counts)
        counts = (captured, uncovered, lost)
        score = sum(counts[i] * weight for i, weight in enumerate(self.heuristic_weights))
        return score

    def get_next_val(self, state, depth, get_max=True, alpha=-1000, beta=1000, tabs=0, root=False):
        this_fn = 'max' if get_max else 'min'
        logger.debug('get_%s_val @ depth: %s, alpha: %s, beta: %s',
                     this_fn, depth, alpha, beta, extra={'tabs': '\t' * tabs})
        prob_match_over, winner = state.is_match_over()
        if prob_match_over == 1.0 and root:
            return None
        if prob_match_over > 0.0:
            match_score = 3 + 9 - sum(state.captures) if winner == 'P' else -sum(state.counts)
            value = 10 * match_score * prob_match_over
            # logger.debug('get_%s_val: Game over possible. Winner: %s, Prob: %s, score: %s, Returning: %s',
            #              this_fn, winner, prob_match_over, match_score, value, extra={'tabs': '\t' * tabs})
            return value
        if depth == 0:
            value = self.get_state_heuristic(state) if not root else None
            # logger.debug('get_%s_val: Depth limit reached. Returning: %s',
            #              this_fn, value, extra={'tabs': '\t' * tabs})
            return value

        return self.get_next_val_inner(state, depth, get_max, alpha, beta, tabs, root)

    def get_next_val_inner(self, state, depth, get_max, alpha, beta, tabs, root):
        # this_fn = 'max' if get_max else 'min'
        # next_fn = 'min' if get_max else 'max'
        comparator = (lambda x, y: x > y) if get_max else (lambda x, y: x < y)
        moves = state.get_possible_moves('P' if get_max else 'O')
        # logger.debug('possible moves: %s', [self.get_move_code(m) for m in moves], extra={'tabs': '\t' * tabs})
        # sort moves by their hash table scores
        state_hash = state.get_hash()
        if state_hash not in self.history_table:
            self.history_table[state_hash] = {}
        move_scores = self.history_table[state_hash]
        moves.sort(key=lambda x: move_scores.get(x, 0), reverse=True)
        # logger.debug('sorted moves:   %s', [self.get_move_code(m) for m in moves], extra={'tabs': '\t' * tabs})
        # logger.debug('move scores:    %s', [move_scores.get(m, 0) for m in moves], extra={'tabs': '\t' * tabs})
        best_move, best_value = None, None
        for move in moves:
            from_ring, from_index = move[0]
            from_piece = state.get_board_value(from_ring, from_index)
            to_ring, to_index = move[1]
            to_piece = state.get_board_value(to_ring, to_index)
            if to_piece == '0':
                # movement
                clone = state.clone()
                clone.apply_move(move[0], move[1], 'M')
                # logger.debug('--> calling %s with movement: %s', next_fn, self.get_move_code(move),
                #              extra={'tabs': '\t' * tabs})
                value = self.get_next_val(clone, depth - 1, not get_max, alpha=alpha, beta=beta, tabs=tabs + 1)
                # logger.debug('<-- called %s with movement: %s, and got %s',
                #              next_fn, self.get_move_code(move), value, extra={'tabs': '\t' * tabs})
            else:
                # challenge
                piece1 = from_piece[1]
                piece2 = to_piece[1]
                opponent_piece = piece2 if get_max else piece1
                if opponent_piece != 'U':
                    # known piece
                    outcome = self.get_challenge_outcome(piece1, piece2)
                    clone = state.clone()
                    clone.apply_move(move[0], move[1], outcome, opponent_piece)
                    # logger.debug('--> calling %s with known challenge: %s', next_fn, self.get_move_code(move),
                    #              extra={'tabs': '\t' * tabs})
                    value = self.get_next_val(clone, depth - 1, not get_max, alpha=alpha, beta=beta,
                                              tabs=tabs + 1)
                    # logger.debug('<-- called %s with known challenge: %s and got %s',
                    #              next_fn, self.get_move_code(move), value, extra={'tabs': '\t' * tabs})
                else:
                    # unknown piece
                    probabilities = state.get_opponent_piece_probabilities()
                    values = [0] * 3
                    interval = (-27, 36)
                    # logger.debug('--> calling %s with unknown challenge: %s, '
                    #              '\n' + ('\t' * tabs) + '\tprobabilities: %s',
                    #              next_fn, self.get_move_code(move), probabilities, extra={'tabs': '\t' * tabs})
                    for i in range(3):
                        if probabilities[i] > 0:
                            if get_max:
                                opponent_piece = piece2 = state.PIECE_KEY[i]
                            else:
                                opponent_piece = piece1 = state.PIECE_KEY[i]
                            outcome = self.get_challenge_outcome(piece1, piece2)
                            # logger.debug('*** chance node, i: %s, assuming: %s, outcome: %s',
                            #              i, opponent_piece, outcome, extra={'tabs': '\t' * tabs})
                            clone = state.clone()
                            clone.apply_move(move[0], move[1], outcome, opponent_piece)
                            values[i] = self.get_next_val(clone, depth - 1, not get_max,
                                                          alpha=alpha, beta=beta, tabs=tabs + 1)
                        if i < 2 and sum(probabilities[:i + 1]) > 0:
                            known = sum([values[j] * probabilities[j] for j in range(i + 1)])
                            bounds = [(known + sum(probabilities[i + 1:]) * interval[j]) for j in range(2)]
                            # logger.debug('*** chance node bounds: %s',
                            #              bounds, extra={'tabs': '\t' * tabs})
                            if (get_max and bounds[0] > beta) or ((not get_max) and bounds[1] < alpha):
                                # logger.debug('*** chance node pruning!', extra={'tabs': '\t' * tabs})
                                values = values[:i + 1]
                                probabilities = [probabilities[j] / sum(probabilities[:i + 1]) for j in range(i + 1)]
                                break
                    value = sum(values[i] * probabilities[i] for i in range(len(values)))
                    # logger.debug('<-- called %s with unknown challenge: %s, and got: %s,'
                    #              '\n' + ('\t' * tabs) + '\tprobabilities: %s, values: %s',
                    #              next_fn, self.get_move_code(move), value, probabilities, values,
                    #              extra={'tabs': '\t' * tabs})
            if best_value is None:
                best_move, best_value = move, value
            if comparator(value, best_value):
                best_move, best_value = move, value
            boundary = beta if get_max else alpha
            if comparator(best_value, boundary) or best_value == boundary:
                # fail high on max or fail low on min
                # logger.debug('get_%s_val: pruning!!! Returning move %s with value %s',
                #              this_fn, self.get_move_code(best_move), best_value, extra={'tabs': '\t' * tabs})
                # add move to history table
                if best_move not in self.history_table[state_hash]:
                    self.history_table[state_hash][best_move] = 0
                self.history_table[state_hash][best_move] += 1
                return best_move if root else best_value
            if get_max and best_value > alpha:
                alpha = best_value
            elif not get_max and best_value < beta:
                beta = best_value
        if root:
            logger.debug('get_max_val: Returning move %s with value %s',
                         self.get_move_code(best_move), best_value, extra={'tabs': '\t' * tabs})
        # add move to history table
        if best_move not in self.history_table[state_hash]:
            self.history_table[state_hash][best_move] = 0
        self.history_table[state_hash][best_move] += 1
        return best_move if root else best_value

    @staticmethod
    def get_challenge_outcome(challenger, defender):
        outcome = None
        if challenger == defender:
            outcome = 'T'
        elif challenger == 'R':
            outcome = 'W' if defender == 'S' else 'L'
        elif challenger == 'P':
            outcome = 'W' if defender == 'R' else 'L'
        elif challenger == 'S':
            outcome = 'W' if defender == 'P' else 'L'
        return outcome
