##  :video_game: Action
    0-51: attack or defense card with index respectively:
        0-12: Two Spade - Ace Spade
        13-25: Two Club - Ace Club
        26-38: Two Diamond - Ace Diamond
        39-51: Two Heart - Ace Heart

    52: skip turn (attacker) or defend fail(defender).


##  :bust_in_silhouette: P_state

    state[0:52] = {"1":have that card,"0":dont have that card} (index of the card)
    state[52:104] = {"1":card defend successful(being defend),"0":not or dont have to}
    state[104:156] = {"1":card have to defend this round}
    state[156:158] = [1,0] ("attack","defense") 
    state[158:162] = [1,0,0,0] (trump suit)
    state[162] = num card left on deck
    state{163:166} = num card left each player
    state[166]: end game


##  :globe_with_meridians: ENV_state

    *Index:(env[0:52])
    0-12: Two Spade - Ace Spade
    13-25: Two Club - Ace Club
    26-38: Two Diamond - Ace Diamond
    39-51: Two Heart - Ace Heart

    *Player Info:
    -Card on hand: 8 card
    -Encode: if player had card, the card will encode with player ID
    +Ex:player 1 has King Spade so index 13 is 1 .[4 1 2 3 1 2 4 1 2 3 1 2 1]
    +If noone have that card(card thrown away when defend successful), encode that card -1
    +If the card still on deck,encode the card 0 ;
    +If the card  have to be defense, encode 5;
    +If the card defended successful, encode from 5 to 6;
    +If not all the card turn into 6; player defend fail, else player defend successful.
    *Trump suit: the master suit of the game (env[52])
    -Mode: Attack or Defense: env[53]
    -Check if player not attack: 0-3  (env[57])
    -A variable to track which player attack: player_id. (env[58])

    *How env change:
    -env[0:52]: value 1,2,3,4 if player hold that card with player_id perspectively,value 0 if card still on the deck, 
                value -1 if card thrown away,
                value 5 if the defend player have to defend that card, 
                value 6 means the card defended successfully and on the defend board,

    -env[52]: Trump suit card, determined the master suit of this game, more valuable than any other suit and value.
    Example: value 12 means K spade is chosen and spade is the master suit of the game.

    -env[53]: mode of the player. {"1":Defense,"0":Main Attack}.

    -env[54:57]: attack player_id (if defend player_id is 2 then env[54:57]=[1,3,4])

    -env[57]: when a player choose attack reset to 0, 
                    defend player defense successful if = 3, 
                    player pass attack to other player +1

    -env[58]: track which player_id is defending.

    -env[59]: index player attack in env[54:57]. (to track which player is attacking)

    -env[60:80]: card on deck (trump card is the last card on the deck) (to draw card)
    -env[80]: end game (0: not end else 1)

