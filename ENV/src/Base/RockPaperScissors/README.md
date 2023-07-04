##  :video_game: Action
    * [0]: Scissors
    * [1]: Rock
    * [2]: Paper
    * [3]: Confirm
    
##  :bust_in_silhouette: P_state
    * [0: 3]: Player's choice
    * [3: 6]: Opponent's choice
    * [6]: Phase
    * [7]: 1 if the game over
##  :globe_with_meridians: ENV_state
    * [0: 2]: The hand shape of each player 
        -values: -1: if no choice has been made yet
                  0: Scissors
                  1: Rock
                  2: Paper
    * [2]: Turn
    * [3]: The player who is currently playing ( 0 or 1)
    * [4]: Phase:
        + 0: Each player makes a choice
        + 1: confirm the information of the game turn
    * [5]: winner ( -1: no winner, 0, 1)
    * [6]: 1 is the game over, if not, it's not

