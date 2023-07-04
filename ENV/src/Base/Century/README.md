## :dart: Báo cáo Ticket To Ride 
1.   `Tốc độ chạy`
      - **1000 Game**: 
      - **1000 Game full numba**: 
      - **10000 Game**: 

2. `Chuẩn form`: **Đã test**
3. `Đúng luật`: **Đã check**
4. `Không bị loop vô hạn`: **Đã test** với 1000000 ván
5. `Tốc độ chạy các hàm con mà người chơi dùng`: 1000game: 
6. `Số ván check_vic > victory_thật`: 
7. 
9. `Tối thiểu số lần truyền vào player`: 

## :globe_with_meridians: env_state
*   [0:255] **Thông tin người chơi**: gồm 5 người chơi với thông tin từng người là (điểm, số thẻ đã mua, 4 vị trí thể hiện số lượng 4 loại token, 45 vị trí cho 45 loại thẻ hành động)
*   [255:309] **Thông tin 6 thẻ action lật trên bàn, mỗi thẻ gồm 9 thuộc tính (#(số lượng tài nguyên bỏ ra(4 vị trí), số lượng tài nguyên nhận(4 vị trí), số lần nâng cấp (1 vị trí)))**
*   [309:329] **Thông tin token free có sẵn ở các thẻ lật trên bàn, chỉ xét 5 thẻ đầu**
*   [329:354] **thông tin  5 thẻ điểm lật trên bàn** mỗi thẻ có 5 vị trí, 4 vị trí đầu là chi phí mua, vị trí sau cùng là điểm
*   [354:397] **chuỗi lưu trữ thứ tự các thẻ action sau khi xáo** 
*   [397:433] **CHuỗi lưu trữ thứ tự các thẻ điểm sau khi xáo** 
*   [433] **số lần người chơi đã nâng cấp khi dùng thẻ nâng cấp** 
*   [434] **Thẻ người chơi định mua hoặc là thẻ người chơi vừa dùng**
*   [435]   **Số token người chơi cần phải bỏ ra**
*   [436]  **Số đồng bạc còn trên bàn chơi**
*   [437]   **Số đồng vàng còn trên bàn chơi**
*   [438]   **Action mà người chơi vừa thực hiện khi dùng action card**
*   [439]   **Phase của game** 
*   [440]   **Trạng thái game kết thúc hay chưa**
*   [441]   **ID Người chơi được hành động**
*   [442:447]   **Điểm khi bắt đầu của người chơi**

**Total env_state length: 447**


## :bust_in_silhouette: P_state
*   [0]     : **điểm của người chơi đang action**
*   [1]     : **số thẻ điểm người chơi đã mua**
*   [2:6]:  : **số lượng 4 loại nguyên liệu của người chơi** (vang, đỏ, xanh, nâu)
*   [6:51]:   **45 vị trí thể hiện các thẻ action mà người chơi có thể dùng** thẻ nào có thì vị trí tương ứng là 1, còn không thì là 0
*   [51:96]:  **5 vị trí thể hiện các thẻ action mà người chơi đã dùng và cần nghỉ ngơi để khôi phục** thẻ nào có thì vị trí tương ứng là 1, còn không thì là 0
=> các vị trí nào ở cả 2 đoạn trên đều bằng 0 thì là các thẻ người chơi chưa có

*   [96:120]:   **Thông tin của 4 người chơi còn lại**(điểm, số thẻ điểm đã mua, số lượng 4 loại token)
*   [120:174]:  **Thông tin 6 thẻ action lật trên bàn, mỗi thẻ gồm 9 thuộc tính**
*   [174:194]:  **Thông tin token free có sẵn ở các thẻ lật trên bàn, chỉ xét 5 thẻ đầu**
*   [194:219]:  **thông tin  5 thẻ điểm lật trên bàn** mỗi thẻ có 5 vị trí, 4 vị trí đầu là chi phí mua, vị trí sau cùng là điểm
*   [219]       **Số đồng bạc còn trên bàn**
*   [220]       **Số đồng vàng còn trên bàn**
*   [221:266]       **Action vừa thực hiện (khi dùng action card)** 45 vị trí ứng với 45 loại action dùng thẻ
*   [266]       **Trạng thái game dừng hay chưa**
*   [267:272]   **Phase của game, 5 vị trí ứng với 5 phase**
*   [272:277]   **Thứ tự bắt đầu game**

**Total player_state length: 277**



## :video_game: ALL_ACTION
* range(0,1):action nghỉ ngơi
* range(1,7): action mua action card
* range(7, 12): action mua thẻ điểm
* range(12, 57): action dùng thẻ hành động (action card)
* range(57, 61): action chọn loại token để bỏ
* range(61,62): action không tiếp tục sử dụng thẻ action
* range(62,65): action chọn tài nguyên để nâng cấp (vàng, đỏ, xanh)

**Total 65 action**