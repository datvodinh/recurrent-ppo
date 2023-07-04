##  :dart: Báo cáo Poker Texas Holdem
1.   `Tốc độ chạy`
      - **1000 Game**: 
      - **1000 Game full numba**: 
      - **10000 Game**: 

2. `Chuẩn form`: **Đã test**
3. `Đúng luật`: **Đã check**
4. `Không bị loop vô hạn`: **Đã test** với 1000000 ván
5. `Tốc độ chạy các hàm con mà người chơi dùng`: 1000game: 
6. `Số ván check_vic > victory_thật`: chạy 10000 ván thì check_victory = check_winner = 
7. `Giá trị state,  
9. `Tối thiểu số lần truyền vào player`: 

##  :globe_with_meridians: ENV_state
*   [0:52] **mảng tạm đại diện các thẻ trong cỗ bài trên bàn, -1 là vị trí không có**: 
*   [52:57] **Các thẻ mở trên bàn ở các turn, có 5 thẻ, ở các turn chưa được mở thì có giá trị -1**
*   [57:66] **Chip của 9 người chơi**

*   [66:75] **lượng chip người chơi đã bỏ ra trong 1 vòng của 1 ván** 
*   [75:84] **tổng lượng chip người chơi đã bỏ ra trong 1 ván chơi** 
*   [84:93] **trạng thái người chơi (còn chơi hay đã bỏ)** 

*   [93:102] **lá bài đầu tiên của mỗi người chơi** 
*   [102:111] **lá bài thứ 2 của mỗi người chơi** 
*   [111:120] **lá bài đầu tiên của mỗi người chơi khi showdown** 
*   [120:129] **lá bài thứ 2 của mỗi người chơi khi showdown (nếu chỉ show 1 lá thì ô này có giá trị -1)**
*   [129] **id của người chơi giữ button**
*   [130] **id của người chơi giữ button tạm (có tác dụng khi raise)**
*   [131] **trạng thái của game**:
                    -0: pre flop
                    - 3: flop
                    - 4: turn
                    - 5: river
                    - 6: showdown
*   [132] **phase game**
*   [133] **id của người chơi đang action**
*   [134] **cash to call old**
*   [135] **cash to call new**
*   [136] **tổng giá trị pot**
*   [137] **số lượng ván đã chơi trong tour**
*   [138] **check end tour**

**Total env_state length: 139**

##  :bust_in_silhouette: P_state
*   [0:468] là **bài của 9 người chơi khác (chỉ thấy các ng khác khi showdown)**
*   [468:477] **tổng chip còn lại của các người chơi** 
*   [477:486]:   **Tổng chip người chơi đã bỏ ra trong ván để theo**
*   [486:495]:   **trạng thái các người chơi còn theo hay fold**
*   [495:504]:   **button dealer, tại ai thì vị trí ng đó là 1**
*   [504]: **cash to call**
*   [505]: **cash to bet**
*   [506]: **tổng giá trị pot**
*   [507]:   **phase game**
*   [508:513]:   **trạng thái ván chơi**
*   [513]: **số ván đã chơi**
*   [514]: **check end**

**Total: player state length = 515**

##  :video_game: ALL_ACTION
    - 0: call
    - 1: check
    - 2: fold
    - 3: bet/raise
    - 4: all in
    - 5: dừng bet

**Total 6 action**

#  PHASE:
    - 0 : call, check, fold, bet, allin,  (0, 1, 2, 3, 4) (chưa ai bet)
    - 1 : bet, allin, dừng bet, (3, 4, 5)                 (đã có người bet)

#  STATUS GAME:
    - 0: pre flop
    - 3: flop
    - 4: turn
    - 5: river
    - 6: showdown























