##   Thông tin quan trọng:
    - Ảnh map: xem ở folder "src/Base/Catan"
    - 203: Điểm của bản thân
    - 393, 578, 763: Điểm của 3 người chơi còn lại
    - Mô tả thông tin map: Ở file env có 7 array mô tả thông tin map:
        + TILE_TILE: Array thứ i mô tả ô đất thứ i cạnh những ô đất nào.
        + POINT_POINT: Mô tả điểm nào cạnh điểm nào.
        + ROAD_POINT: Mô tả con đường nào được tạo thành từ 2 điểm nào.
        + POINT_ROAD: Mô tả điểm là một đầu mút của những đường nào.
        + PORT_POINT: Mô tả cảng nào được cấu thành từ 2 điểm nào.
        + POINT_TILE: Mô tả điểm nào cạnh ô đất nào.
        + TILE_POINT: Mô tả ô đất nào được cấu thành từ 6 điểm nào.
    Tất cả đều ngoại trừ các giá trị -1.


##  :video_game: Action
    * 0-53: Chọn điểm
    * 54: Roll xúc xắc
    * 55,56,57,58: Dùng thẻ dev, lần lượt Knight, RoadBuilding, Yearofplenty, Monopoly
    * 59,60,61,62,63: Các action lấy nguyên liệu
    * 95,96,97,98,99: Các action trả nguyên liệu
    * 64-82: Action chọn ô để di chuyển Robber
    * 83,84,85: Action chọn cướp tài nguyên của ai
    * 100,101,102: Action chọn giao dịch với ai
    * 86: Mua đường, 87: Mua nhà, 88: Mua thành phố, 89: Mua thẻ dev, 90: Trade người, * 91: Trade bank
    * 92: Kết thúc lượt
    * 93: Không đồng ý trade
    * 94: Đồng ý trade
    * 103: Dừng đưa ra tài nguyên khi trading
    * 104: Dừng yêu cầu tài nguyên khi trading
    * 105: Người chơi chính không trade với ai cả

##  :bust_in_silhouette: P_state
    * [0:95] 5 vị trí liên tiếp là trạng thái nguyên liệu trên một ô đất (tổng 19 ô đất), 0 là không có, 1 là có, nguyên liệu được sắp xếp theo Cây Gạch Cừu Lúa Đá

    * [95:114] Vị trí Robber

    * [114:133] Ô đất nào có value = 2

    * [133:187] 6 vị trí liên tiếp là trạng thái cảng trên một cảng (tổng 9 cảng), thứ tự là Cây Gạch Cừu Lúa Đá, cuối cùng là cảng 3:1

    * [187:192] Tài nguyên trong ngân hàng (còn hay không), thứ tự Cây Gạch Cừu Lúa Đá

    * [192] Ngân hàng còn thẻ phát triển hay không

    * [193:198]: Nguyên liệu bản thân

    * [198:203]: Thẻ phát triển của bản thân

    * [203]: Điểm của bản thân

    * [204:276]: Trạng thái đường của bản thân, có hoặc không có đường tại các vị trí trên bản đồ

    * [276:330]: Trạng thái nhà của bản thân

    * [330:384]: Trạng thái thành phố của bản thân

    * [384]: Số thẻ knight bản thân đã dùng

    * [385]: Độ dài con đường dài nhất của bản thân

    * [386:391]: Tỉ lệ trao đổi với ngân hàng của bản thân

    * [391+185*(i-1):391+185*i], i = 1, 2, 3 là thông tin của 3 người chơi còn lại, một cụm thông tin có độ dài 185, bao gồm các thông tin sau:
        - [0]: Tổng số tài nguyên
        - [1]: Tổng số thẻ phát triển
        - [2]: Số điểm
        - [3:75]: Trạng thái đường
        - [75:129]: Trạng thái nhà
        - [129:183]: Trạng thái thành phố
        - [183]: Số thẻ knight đã dùng
        - [184]: Độ dài con đường dài nhất

    * [946]: Tổng số xúc xắc

    * [947:963]: Phase, lần lượt là
        - 11: Yêu cầu tài nguyên khi trade với người
        - 6: Chọn các mô đun giữa turn
        - 10: Đưa ra tài nguyên khi trade với người
        - 3: Trả tài nguyên do bị chia bài
        - 12: Người chơi phụ phản hồi trade
        - 15: Chọn tài nguyên muốn nhận từ ngân hàng
        - 14: Chọn tài nguyên khi trade với ngân hàng
        - 1: Chọn các điểm đầu mút của đường
        - 4: Di chuyển Robber
        - 13: Người chơi chính duyệt trade
        - 5: Chọn người để cướp tài nguyên
        - 2: Đổ xúc xắc hoặc dùng thẻ dev
        - 0: Chọn điểm đặt nhà đầu game
        - 8: Chọn các điểm mua nhà
        - 7: Chọn tài nguyên khi dùng thẻ dev
        - 9: Chọn các điểm mua thành phố

    * [963:1017]: Điểm đặt thứ nhất (trong phase xây đường)

    * [1017]: Số tài nguyên phải bỏ do bị chia

    * [1018:1022]: Thẻ phát triển đang dùng là gì

    * [1022]: Số lần dùng thẻ phát triển còn lại

    * [1023:1027]: Loại thẻ dev được dùng trong turn hiện tại

    * [1027]: Số lần được tạo trade offer

    * [1028:1033]: Lượng tài nguyên đưa ra trong trade offer

    * [1033:1038]: Lượng tài nguyên yêu cầu trong trade offer

    * [1038:1044]: Người chơi phụ phản hồi trade offer

    * [1044:1047]: Người chơi chính là ai (nếu là bản thân thì = 0 hết)

    * [1047]: Đánh dấu game đã kết thúc

    * [1048+19*v:1048+19(v+1)] Ô đất nào có value = k[v], k = [3, 4, 5, 6, 8, 9, 10, 11, 12], v = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    * [1219:1224]: Giá trị xúc xắc là gì (2, 3, 4, 5, 6)

    * [1224:1229]: Giá trị xúc xắc là gì (8, 9, 10, 11, 12)

