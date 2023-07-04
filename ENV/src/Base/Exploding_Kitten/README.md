## :video_game: Action

    Action index: 10 action
    Phase 1: Main turn
        Action 0-9:
        Action 0: Nope
        Action 1: Attack
        ACtion 2: Skip
        Action 3: Favor
        Action 4: Shuffle
        Action 5: See the future
        Action 6: Draw card(end turn).
        Action 7: Steal a random card from other player (two of a kind).
        Action 8: Name the card you want from other player(three of a kind).
        Action 9: take any card from Discard Pile (5 different cards).

    Phase 2: Nope turn.
        Action 0: Nope
        Action 10: Not Nope


    Phase 3: Steal some card
            Action 11 to 15:  choose player to take.
    Phase 4: choose card to give / take
        Action 3: 
            Action 15 to 27: player been chosen choose a card to give..
        Action 8:
            Action 27 to 39: ask the card you want from other player
        Action 9:
            Action 39 to 51: choose card from Discard Pile

    Phase 5: Discard phase:
        Action 51 to 62: Discard card (index respect to card.txt).



## :bust_in_silhouette: P_state

    state[0:12]:number of card player hold:
        [0]: Nope
        [1]: Attack
        [2]: Skip
        [3]: Favor
        [4]: Shuffle
        [5]: See the future
        [6]: TACOCAT
        [7]: RAINBOW-RALPHING CAT
        [8]: BEARD CAT
        [9]: HAIRY POTATO CAT
        [10]: CATERMELON
        [11]: Defuse

    state[12:25]:number of card in Discard Pile (same index as state[0:12],state[24]:Exploding card)
    state[25]: number of card left in the Draw Pile
    state[26]: number of player left
    state[27]: 1 if player card is Nope by other player else 0
    state[28:41]:first card (See the future)
    state[41:54]:second card (See the future)
    state[54:67]:third card (See the future)
    state[67:71]:  {main turn,nope turn,steal turn, choose/take card turn} (discard phase if sum(state[67:71])==0)
    state[71]: number of card player have to draw
    state[72:82]: main player last action.
    state[82:86]: other player lose or not(1 if not lose else 0)
    state[86]: Exploding (0 if explode else 1)
    state[87:91]: number of card other player have (0 if lose or dont have card)
    state[91:102]: card that have been discard
    state[102]: number of card player have to discard



## :globe_with_meridians: ENV_state
        env[0:56]:
            0,1,2,3,4: id of player   
            5:card on Draw Pile
            6: card on Discard Pile
        DrawPile: all card on Draw Pile with id. (len 31 - 0)
        DiscardPile: all card on DiscardPile with id (len 13: 13 type)

        env[56]: nope count(1 if other player use nope else 0)
        env[57]: track player's main turn (0 to 4)
        env[58:62]: track player's Nope turn (if is player 1 main turn then Nope turn is [2,3,4,0])
        env[62:67]: check if player lose or not (1 if not lose else 0 if lose, default [1,1,1,1,1])
        env[67]: check phase (0,1,2,3)
        env[68]: number of card player env[57] have to draw
        env[69:72]: three card (See the future): id if player use else 0
        env[72]: env[57] last action(track in nope turn)
        env[73]: player id in Nope turn (Nope phase)
        env[74]: player id chosen in phase 2 (steal card turn)
        env[75]: num card main player have to discard
        env[76:87]: card main player have been discard (len 11)
        
