##   Thông tin quan trọng:
    -   142: Điểm của bản thân
    -   186, 230, 274: Điểm của 3 người chơi còn lại
    
##  :video_game: Action
    -    0: Dừng lấy công cụ
    -    [1:11]: Đặt số người (từ 1 đến 10)
    -    11: Đặt người vào lúa
    -    12: Đặt vào ô công cụ
    -    13: Đặt vào ô sinh sản
    -    14: Đặt vào khu gỗ
    -    15: Đặt vào khu gạch
    -    16: Đặt vào khu bạc
    -    17: Đặt vào khu vàng
    -    18: Đặt vào khu lương thực
    -    [19:23]: Đặt vào thẻ civ (0 -> 4)
    -    [23:27]: Đặt vào thẻ build (0 -> 4)
    -    [27]: Chọn trừ nguyên liệu (Khi đến hết vòng không đủ thức ăn)
    -    [28]: Chọn trừ điểm (Khi đến hết vòng không đủ thức ăn)
    -    [29:37]: Lấy người từ lúa, cụ, sinh, gỗ, gạch, bạc, vàng, lương thực
    -    [37:40]: Dùng công cụ, 1,2,3 (Thứ tự ở các ô công cụ)
    -    [40:44]: Trả nguyên liêu, gỗ, gạch, bạc, vàng
    -    [44:47]: Dùng thẻ công cụ 1 lần 3 loại 
    -    [47]: Dừng trả nguyên liệu khi mua thẻ build 1-7 (Thẻ build trả 1 đến 7 người để đổi ra điểm)
    -    [48:52]: Lấy người từ thẻ civ (0 -> 4)
    -    [52:56]: Lấy người từ thẻ build (0 -> 4)
    -    [57:63]: chọn xúc xắc từ 1 đến 6 (Thẻ civ chọn số xúc xắc để lấy nguyên liệu hoặc công cụ hoặc lúa)
    -    [63]: Chọn dùng thẻ lấy thêm 2 nguyên liệu từ thẻ civ
    -    [64:68]: Lấy nguyên liệu gỗ gạch, bạc, vàng
  
  
  
  ##  :bust_in_silhouette: P_state
    -    Thứ tự nguyên liệu: Gỗ, Gạch, Bạc Vàng
    -    thứ tự các ô: Lúa, cụ, sinh, gỗ, gạch, bạc, vàng, thực, 4 thẻ civ, 4 thẻ build

    -    [0:4]: Số nguyên liệu còn lại của mỗi ô
    -    [4:8]: số thẻ Build còn lại ở mỗi ô
    -    [8:12]: Số nguyên liệu đã trả trong turn trả nguyên liệu khi mua thẻ civ hoặc build
    -    [12]: Số lượng nguyên liệu cần phải trả khi trả thay thức ăn
    -    [13]: GAME ĐÃ END HAY CHƯA
    -    [14:110]: thông tin của các thẻ civ trên bàn
    -    [110:142]: Thông tin các thẻ build trên bàn
    -    [142:318]: Thông tin của các người chơi [142:142+44*4]
    -        0: Điểm 
    -        1: Lúa
    -        2: Số người
    -        3: Số lương thực
    -        4: Số nhà
    -        [5:9]: Nguyên liệu đang có
    -        [9:12]: Thẻ công cụ đang có (Trong các ô công cụ)
    -        [12:15]: 3 thẻ dùng công cụ 1 lần
    -        [15:18]: State của 3 thẻ dùng công cụ
    -        [21]: Đã trả thức ăn
    -        [22:30]: 8 thẻ văn minh các loại(Người, sáo, cây, dệt, đồng hồ, lu, xe, đá)
    -        [30]: Có thể dùng thẻ lấy 2 nguyên liệu 1 lần.
    -        [31:39]: Số người có của mỗi người ở mỗi ô (Lúa, cụ, sinh, gỗ, gạch, bạc, vàng, thực)
    -        [39:43]: số người Tính điểm cuối game khi mua thẻ civ(lúa, nhà, công cụ, số người)
    -    [318:322]: Người đang chơi (tạm bỏ)
    -    [322:326]: Người đang ở có ở các ô civ, cũng là số nguyên liệu cần phải trả
    -    [326:330]: Người đang có ở các ô build
    -    [330:338]: Thông tin card của người chơi 1(theo góc nhìn của agent)
    -    [338:346]: Thông tin card của người chơi 2(theo góc nhìn của agent)
    -    [346:354]: Thông tin card của người chơi 3(theo góc nhìn của agent)

    -    [354: 370]: Ô vừa chọn để đặt người (push)
    -    [370: 386]: Ô vừa chọn để lấy người (pull)
    -    [386]: Tổng công cụ đã dùng

    -    [387:411]: giá trị 4 xúc xắc của 4 người khi dùng thẻ civ roll xúc xắc, mỗi xúc xắc có 6 index
    -    [411]: Tổng giá trị xúc xắc
    -    [412:423]: Phase (có 11 phase)
    -        0:   Chọn ô đặt người
    -        1:   Đặt số người
    -        2:   Lấy người từ các ô
    -        3:   Trả nguyên liệu mua thẻ civ
    -        4:   Dùng công cụ(end hoặc hết thì qua roll xúc xắc)
    -        5:   trả nguyên liệu khi mua thẻ build 1 -> 7
    -        6:   Chọn trừ nguyên liệu hoặc trừ điểm khi không đủ thức ăn
    -        7:   Chọn giá trị xúc xắc khi dùng thẻ civ xúc xắc
    -        8:   Lấy nguyên liệu từ ngân hàng khi dùng thẻ civ lấy 2 nguyên liệu bất kỳ
    -        9:   trả nguyên liệu khi mua thẻ civ có số lượng mặc định
    -        10:   trả nguyên liệu nếu không đủ thức ăn
  
  
##  Civ card
    -   0: Loại thẻ có dùng xúc xắc hay không(thẻ Roll xúc xắc lấy giá trị rồi chia cho nguyên liệu trong thẻ)
    -   1: Thẻ roll xúc xắc(Thẻ roll 4 xúc xắc, mõi người chọn một loại tài nguyên tương ứng với số xúc xắc)
    -   2: Thẻ lấy thức ăn
    -   3: thẻ lấy điểm
    -   4: Thẻ lấy 1 gỗ
    -   5: Thẻ lấy 1 hoặc 2 bạc(3 thẻ, 2 thẻ lấy 1. 1 thẻ lấy 2)
    -   6: Thẻ lấy 1 vàng
    -   7: Thẻ dùng công cụ(4 thẻ: 1, 2, 3, 4)
    -   8: Thẻ có thêm 1 lúa
    -   9: Thẻ có thêm 2 nguyên liệu bất kỳ
    -   10: Thẻ nâng cấp nền văn minh(Thẻ này lấy thêm một thẻ civ khác, nhưng chỉ dùng phần tính điểm cuối game)
  
    -   11: Thẻ nền xanh: hình người
    -   12: Thẻ nền xanh: sáo
    -   13: Thẻ nền xanh: cây
    -   14: Thẻ nền xanh: khung dệt
    -   15: Thẻ nền xanh: đồng hồ đá
    -   16: Thẻ nền xanh: cái lu
    -   17: Thẻ nền xanh: xe đẩy
    -   18: Thẻ nền xanh: cục đá
  
    -   19: Thẻ nền vàng: Số nguời
    -   20: Loại nâng cấp: Nhà
    -   21: Loại nâng cấp: công cụ
    -   22: Loại nâng cấp: số người
    -   23: Loại nâng cấp: lúa
       
  
##  Building card
    -   0: Điểm
    -   1: Gỗ
    -   2: Gạch
    -   3: Bạc
    -   4: Vàng
    -   5: Thẻ đổi điểm bằng nguyên liệu từ 1-7
    -   6: Thẻ đổi điểm bằng nguyên liệu từ 4 hoặc 5
    -   7: Số lượng nguyên liệu khác nhau
  
##  :globe_with_meridians: ENV_state
    - [0:4]: Nguời đang chơi
    - 4: Vòng chơi
    - [5:41]: 36 thẻ Civ
    - [41:69]: 28 thẻ build
    - [69:73]: Số nguyên liệu còn lại của ngân hàng
    - [73:77]: số thẻ Build còn lại ở mỗi ô
    - [77:81]: Số nguyên liệu đã trả trong turn trả nguyên liệu khi mua thẻ civ hoặc build
    - [81]: Số lượng nguyên liệu cần phải trả khi trả thay thức ăn
    - [82]: GAME ĐÃ END HAY CHƯA
    - [83:179]: thông tin của các thẻ civ
    - [179:211]: Thông tin các thẻ build
    - [211:371]: Thông tin của các người chơi [211:211+44*4]
       -    0: Điểm
       -    1: Lúa
       -    2: Số người
       -    3: Số lương thực
       -    4: Số nhà
       -    [5:9]: Nguyên liệu đang có
       -    [9:12]: Thẻ công cụ đang có
       -    [12:15]: 3 thẻ dùng công cụ 1 lần
       -    [15:18]: State của 3 thẻ dùng công cụ
       -    [21]: Đã trả thức ăn
       -    [22:30]: 8 thẻ văn minh các loại(Người, sáo, cây, dệt, đồng hồ, lu, xe, đá)
       -    [30]: Có thể dùng thẻ lấy 2 nguyên liệu 1 lần
       -    [31:39]: Số người có của mỗi người ở mỗi ô (Lúa, cụ, sinh, gỗ, gạch, bạc, vàng, thực)
       -    [39:43]: số người Tính điểm cuối game (lúa, nhà, công cụ, số người)
    - [387:391]: Người đang ở có ở các ô civ, cũng là số nguyên liệu cần phải trả
    - [391:395]: Người đang có ở các ô build
    - [395]: Ô vừa chọn để đặt người
    - [396]: Ô vừa chọn để lấy người
    - [397]: Tổng công cụ đã dùng
    - [398:402]: giá trị 4 xúc xắc của 4 người khi dùng thẻ civ roll xúc xắc
    - [402]: Tổng giá trị xúc xắc
    - [403:]: Phase 