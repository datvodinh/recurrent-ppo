##  :video_game: ALL_ACTION
* range(0,101): action xây đường
* range(101, 147): action drop thẻ route
* range(147, 157): action nhặt thẻ train_car
* range(157, 166): action chọn loại train_car xây road
* range(166, 168): action chọn xây tunnel hay không
* range(168, 170): action nhặt thẻ route và dừng drop thẻ route
* range(170, 171): action skip
**Total 171 action**

##  :bust_in_silhouette: P_state
*   [0:5] là **điểm của các người chơi**
*   [5:10] là **điểm cần trừ của các người chơi** : do ko hoàn thành thẻ route

*   [10:19] **số lượng từng loại thẻ train_car của người chơi**
*   [19:524]:   **đường của 5 người chơi** mỗi người chơi có 101 vị trí ứng với 101 con đường, có đường nào thì vị trí đường đó có giá trị 1
*   [524:570]:   **thẻ route của người chơi** thẻ nào có thì vị trí tương ứng là 1, còn không thì là 0
*   [570:616]:  **thẻ route người chơi lấy trong turn** thẻ nào có thì vị trí tương ứng là 1, còn không thì là 0
*   [616:625]:   **số lượng thẻ train_car mở trên bàn chơi**
*   [625:634]:  **số lượng từng loại thẻ người chơi dùng xây hầm** trong phase lấy nguyên liệu
*   [634:643]: **thẻ bàn chơi bỏ ra để thách người chơi xây hầm**
*   [643:652]: **Các kiểu xây đường có thể**
*   [652]       **ID action (luôn = 0)**
*   [653:657]: phase của bàn chơi, vị trí ứng với các phase lần lượt là 0-1; 1-2; 2-3; 3-4
*   [657]: **Có lấy được thẻ tuyến đường hay không**
*   [658]: **số thẻ route đã bỏ trong lượt**
*   [659]: **game sắp kết thúc chưa**
*   [660:665]: **vị trí người chơi hành động cuối cùng**
*   [665]:  **số thẻ train_car đã lấy**
*   [666]:  **có láy được thẻ train_car úp không**
*   [667]:  **số tàu người chơi còn**
*   [668:673]: **số thẻ route mà người chơi hoàn thành**
*   [673:678]: **người chơi có con đường dài nhất hay không **

**Total: player state length = 678**

##  :globe_with_meridians: ENV_state
*   [0:101] **các đường trên bàn**: range(0,5) là của người chơi, chưa bị sở hữu là -1
*   [101:147] **mảng tạm đại diện thẻ route trên bàn chơi, vị trí nào không có thì là -1, số vị trí khác -1 là số thẻ route còn trên bàn**
*   [147:257] **Mảng tạm đại diện cho các thẻ train_car trên bàn chơi, vị trí không có là -1**, Các giá trị là range(0,9) và -1
*   [257:547] **THuộc tính của 5 người chơi** (ATTRIBUTE_PLAYER = 58       #(score, neg_score, number_train, 9 vị trí cho số lượng thẻ traincar mỗi loại, 46 vị trí cho thẻ route đang có (0 là ko có, 1 là đang giữ, -1 là đã drop))
*   [547:552] **Các thẻ train_car lật trên bàn** cấp 1, 2, 3

*   [552:598] **Các thẻ route người chơi được bỏ trong lượt** thẻ nào có thì vị trí tương ứng là 1, không thì là 0
*   [598:607] **Số lượng từng loại thẻ train_car ở chồng bài bỏ**
*   [607:616]   **Số lượng từng loại thẻ người chơi bỏ ra xây hầm**
*   [616:625]  **Số lượng từng loại thẻ bàn chơi lật ra để thách người chơi xây hầm**
*   [625]   **phase của bàn chơi**
*   [626]   **index người chơi hành động**
*   [627]   **kiểm tra game sắp kết thúc chưa** nếu có người còn dưới 3 tàu thì có giá trị 1
*   [628]   **index người cuối cùng được hành động**
*   [629]   **số thẻ train_car người chơi đã lấy trong lượt**
*   [630:639]   **các loại train_car người chơi có thể dùng xây road**
*   [639]       **Con đường người chơi xây trong lượt**
*   [640]   **Số thẻ route người chơi đã bỏ trong lượt**
*   [641]   **turn của bàn chơi** dùng để xét ở đầu game khi những người chơi bỏ thẻ route
*   [642:647]   **số thẻ route người chơi hoàn thành**
*   [647:652]   **Người chơi có phải là người có con đường dài nhất không, nếu có thì là 1**

**Total env_state length: 652**

