##   Thông tin quan trọng
    14: Điểm của bản thân
    29, 44, 59: Điểm của người chơi khác
    
##  :video_game: Action
    * [0]: Bốc 1 lá từ bộ bài.
    * [1:4]: Chọn người chơi khác (là người bị yêu cầu).
    * [4: 17]: Chọn lá bài yêu cầu ( lá bài = action - 4). ví dụ: Muốn yêu cầu lá số 0 thì là action 4, 
    
##  :bust_in_silhouette: P_state
    * [0:13]: Những lá bài của bản thân. Số lượng lá k = state[k]
    * [13]: Số lượng lá trên tay của bản thân.
    * [14]: Điểm của bản thân
    * [15 + i*15: 30 + i*15], i = 0, 1, 2: Thông tin của người chơi khác:
        - [0: 13] Những bộ 4 ( 1 là có)
        - [13] Số lá trên tay
        - [14] Điểm
    * [60]: Số lá còn lại trên bàn.
    * [61: 64]: Thông tin về PHASE của người chơi:
        - 0: Chọn người chơi khác để yêu cầu
        - 1: Chọn lá bài muốn yêu cầu
        - 2: Bốc
    * [64: 67]: Người bị yêu cầu.
    * [67 + 13*i: 80 + 13*i], i = 0, 1, 2: Những lá bài của người chơi khác đã yêu cầu trong turn gần nhất của họ
    * [106]: Game đã kết thúc hay chưa (1 là kết thúc rồi).
##  :globe_with_meridians: ENV_state
    * [0: 52]: Thứ tự các lá trên bộ bài
    * [52]: Số lá còn lại trên bộ bài bốc ---> Lá bài trên cùng để bốc = env[ 52 - env[52]]
    * [53 + i*15: 68 + i*15]: Thông tin của các người chơi:
        - [0: 13]: Những lá bài của người chơi.
        - [13: 15]: Số lượng lá trên tay và số điểm.
    * [113]: turn
    * [114]: phase ( có 4 phase: 0 là chọn người bị yêu cầu, 1 là chọn lá bài yêu cầu, 2 là Bốc, 3 là chuyển sang người chơi khác)
    * [115]: người đang bị yêu cầu ( 0 là chưa chọn ai, 1-3 )
    * [116]: Lá bài yêu cầu (-1 là chưa yêu cầu lá nào, 0-12)
    * [117 + 13*i: 130 + 13*i], i = 0, 1, 2, 3: Những lá bài những người chơi đã yêu cầu trong turn gần nhất của họ
    * [169]: EndGame
    
#  Luật bổ sung
    Khi những người chơi bằng điểm nhau thì người có bộ cao nhất sẽ là người chiến thắng.
