
##  Luật chơi game này:
    -File: src/Base/CatanNoExchange/HeXSaiGon-Quarantine.pdf
    -Map: src/Base/CatanNoExchange/catan-Page-4.drawio.png

##  :video_game: Action
    * 0-53: Chọn điểm
    * 54: Roll xúc xắc
    * 55,56,57,58: Dùng thẻ dev, lần lượt Knight, RoadBuilding, Yearofplenty, Monopoly
    * 59,60,61,62,63: Các action lấy nguyên liệu
    * 64-82: Action chọn ô để di chuyển Robber
    * 83: build_road
    * 84: build_settlement
    * 85: build_city
    * 86: buy_dev
    * 87: trade_bank
    * 88: pass turn
    * 89, 90, 91, 92, 93: Các action trả nguyên liệu
    * 94: Lấy nguyên liệu từ kho dự trữ cá nhân



##  :bust_in_silhouette: P_state

        19 ô đất, 54 điểm đặt, 72 đường đường, 9 cảng
        Thứ tự tài nguyên: Cây, Gạch, Cừu, Lúa, Đá, Sa mạc
        Thứ tự thẻ dev: KNIGHT, BUILD ROAD, YEAR_OF_PLENTY, MONOPOLY, VICTORY_POINT

    -   [0:114] Tài nguyên trên các ô đất mỗi ô 6 index nguyên liệu (0, 1) (19*6)
    -   [114:133] Vị trí Robber 19 vị trí đặt (0, 1)
    -   [133:361]   Số trên các ô đất (0, 1) (2 -> 12)*19

    -   [361:415]   Các cảng (Cây, Gạch, Cừu, Lúa, Đá, 3:1) 9 cảng *6
    -   [415:420]   Tài nguyên ngân hàng Dạng 0, 1
    -   [420]   Thẻ dev bank Dạng 0, 1
    -   [421:629]  Thông tin cá nhân(Index phía dưới +421)
        - [0:5]: Tài nguyên
        - [5:10]: Thẻ dev
        - [10]: Điểm của người chơi
        - [11:83]: Đường: 72 đường, (0, 1)
        - [83:137]: Nhà: 54 điểm đặt nhà (0, 1)
        - [137:191]: Thành phố: 54 điểm đặt(0, 1)
        - [191]: Số thẻ knight đã dùng (sl)
        - [192]: Con đường dài nhất (sl)
        - [193:208]: Tỉ lệ trao đổi với Bank(với mỗi nguyên liệu, lần lượt là tỉ lệ 2, 3, 4. có 5 nguyên liệu) (0, 1)

    * Thông tin người chơi khác: [629:814], [814:999], [999:1184]
        -  [0]: Tổng tài nguyên(sl)
        -  [1]: Tổng số thẻ dev(sl)
        -  [2]: Điểm (sl)
        -  [3:75]: Đường
        -  [75:129]: Nhà
        -  [129:183]: Thành phố
        -  [183]: Số thẻ knight đã dùng
        -  [184]: Con đường dài nhất

    -   [1184:1188]: Danh hiệu quân đội mạnh nhất (0, 1) (4 người, người đang chơi là index 0)
    -   [1188:1192]: Danh hiệu con đường dài nhất (0, 1) (4 người, người đang chơi là index 0)
    -   [1192:1203]: Tổng xúc xắc (0, 1) (2 -> 12)
    -   [1203]: Nguyên liệu còn lại có thể lấy ở đầu game (sl)
    -   [1204:1258]: Điểm đặt thứ nhất khi đặt đường
    -   [1258]: Số tài nguyên phải bỏ do bị chia(Do đổ xúc xắc ra 7)
    -   [1259:1263]: Đang dùng thẻ dev gì
    -   [1263]: Số lần dùng thẻ dev (Số lần đã dùng trong lượt của mình)
    -   [1264:1268]: Loại thẻ dev được sử dụng trong turn hiện tại
    -   [1268:1273]: Số nguyên liệu còn lại ở trong kho người chơi
    -   [1273:1286]: Các phase, gồm 13 phase 0 -> 12, phase 12 là phase chọn lấy nguyên liệu từ kho
            CÁC PHASE: 
            - 0: Chọn điểm đặt nhà đầu game
            - 1: Chọn các điểm đầu mút của đường
            - 2: Đổ xx hoặc dùng thẻ dev
            - 3: Trả tài nguyên do bị chia bài
            - 4: Di chuyển Robber
            - 5: Lấy nguyên liệu đầu game
            - 6: Chọn các mô đun giữa turn
            - 7: Chọn tài nguyên khi dùng thẻ dev
            - 8: Chọn các điểm mua nhà
            - 9: Chọn các điểm mua thành phố
            - 10: Chọn tài nguyên khi trade với ngân hàng
            - 11: Chọn tài nguyên muốn nhận từ ngân hàng
            - 12: Lấy nguyên liệu từ kho ở trước hoặc sau khi đổ xúc xắc
    
    -   [1286:1291]: Tài nguyên đưa ra trong trade offer để trade với bank
    -   [1291]: EndGame


##  :globe_with_meridians: ENV_state

        -   [0:19] Tài nguyên trên các ô đất
        -   [19] Vị trí Robber
        -   [20:39]   Số trên các ô đất
        -   [39:48]   Các cảng
        -   [48:53]   Tài nguyên ngân hàng
        -   [53:58]   Thẻ dev bank
        -   [58:100]  Thông tin người chơi 0

            - [0:5] Tài nguyên
            - [5:10] Tài nguyên
            - [10] Điểm
            - [11:26] Tài nguyên
            - [26:31] Tài nguyên
            - [31:35] Tài nguyên
            - [35] Số thẻ Knight đã dùng
            - [36] Con đường dài nhất 
            - [37:42] Tỉ lệ trao đổi với bank

        -   [100:142] Thông tin người chơi 1
        -   [142:184] Thông tin người chơi 2
        -   [184:226] Thông tin người chơi 3

        -   [226] Danh hiệu quân đội mạnh nhất
        -   [227] Danh hiệu con đường dài nhất
        -   [228] Tổng xúc xắc
        -   [229] Phase
        -   [230] Turn
        -   [231] Điểm đặt thứ nhất
        -   [232] Số tài nguyên trả do bị chia
        -   [233] Đang dùng thẻ dev gì
        -   [234] Số lần sử dụng thẻ dev

        -   [235:239] Loại thẻ dev được sử dụng trong turn hiện tại
        -   [239:244] Lượng nguyên liệu còn lại khi bị chia đầu game
        -   [244] Người chơi đang action(không hẳn là người chơi chính)
        -   [245:249] Số nguyên liệu đã lấy trong turn đầu game
        -   [249:254] Tài nguyên đưa ra trong trade offer
        -   [254:259] Tài nguyên yêu cầu trong trade offer
        -   [254:274] Tài nguyên trong kho dự trữ của người chơi
        -   [184:226] Thông tin người chơi 3
        -   [280] End Game

#  Thông tin khác
        Số thẻ dev và tài nguyên mỗi loại
        KNIGHT: 14
        BUILD ROAD* 2
        YEAR_OF_PLENTY * 2
        MONOPOLY * 2
        VICTORY_POINT * 5
        DESERT: 1
        lumber* 4
        brick * 3
        sheep * 4
        grain * 4
        ore * 3