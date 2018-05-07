from card_utils import gen_cards, estimate_hole_card_win_rate,estimate_hole_card_draw_rate

def convert_card_to_nums(card):
  
    suits = { #starts from 1 because 0 is the default padding.#
      "D": 1,
      "C": 2,
      "H":3,
      "S": 4,
      }
    faces = {
      "A":14,
      "K":13,
      "Q":12,
      "J":11,
      "T":10,
      "9":9,
      "8":8,
      "7":7,
      "6":6,
      "5":5,
      "4":4,
      "3":3,
      "2":2
      }
    return [suits[card[0]], faces[card[1]]]

def convert_all_cards(cards):
    converted = [convert_card_to_nums(card) for card in cards]
    converted = [[card[0]-1,card[1]-2] for card in converted]
    card_nums = [card[1]*4 + card[0] for card in converted]
    one_hot = [0 for x in range(52)]
    for card_num in card_nums:
        one_hot[card_num] = 1

    return one_hot

def get_bot_info(valid_actions, hole_card, round_state,player_index): #may need sb amount
    info = []
    #money
    info.extend([seat["stack"] for seat in round_state["seats"]]) #each seat stack
    info.append(round_state["pot"]["main"]["amount"]) #pot

    #player positions
    info.append(round_state["dealer_btn"]) #dealer button location
    info.append(round_state["next_player"])#who is acting next? not important for this, but generalizable
    info.append(player_index) #for now, the position of the bot

    #actions
    info.append(valid_actions[1]["amount"]) #amount to call
    info.append(valid_actions[2]["amount"]["min"]) #minimum bet/raise.
    info.append(valid_actions[2]["amount"]["max"]) #amount of money at stake for all-in

    #montecarlo simulations        
    win_rate = estimate_hole_card_win_rate(50, 2,gen_cards(hole_card),gen_cards(round_state["community_card"]))
    draw_rates = estimate_hole_card_draw_rate(100,gen_cards(hole_card),gen_cards(round_state["community_card"]))
    info.append(win_rate)       
    info.extend(draw_rates)
    
    #cards
    info.extend(convert_all_cards(hole_card))
    info.extend(convert_all_cards(round_state["community_card"]))

    return info
    
