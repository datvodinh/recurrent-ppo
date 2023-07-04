##   Thông tin quan trọng:
      16-17-18-19: Thông tin 4 thẻ chiến thắng của bản thân
      [36:40], [56:60], [76:80]: Thông tin 4 thẻ chiến thắng của các người chơi khác
      Total: player state length = 130
      
##  :video_game: ALL_ACTION
      0	Không đổ lại xúc sắc
      1	Đổ 1 xúc sắc
      2	Đổ 2 xúc sắc
      3	Lấy tiền người đầu tiên sau mình
      4	Lấy tiền người thứ 2 sau mình
      5	Lấy tiền người thứ 3 sau mình
      6	Đối thẻ người đầu tiên sau mình
      7	Đổi thẻ người thứ 2 sau mình
      8	Đổi thẻ người thứ 3 sau mình
      9	Không đổi thẻ
      10	Đổi thẻ lúa mì
      11	Đổi thẻ nông trại
      12	Đổi thẻ tiệm bánh
      13	Đối thẻ quán cà phê
      14	Đổi thẻ cửa hàng tiện lợi
      15	Đổi thẻ rừng
      16	Đổi thẻ nhà máy pho mát
      17	Đổi thẻ nhà máy nội thất
      18	Đổi thẻ mỏ quặng
      19	Đổi thẻ quán ăn gia đình
      20	Đổi thẻ vườn táo
      21	Đổi thẻ chợ trái cây
      22	Chọn lấy thẻ lúa mì
      23	Chọn lấy thẻ nông trại
      24	Chọn lấy thẻ tiệm bánh
      25	Chọn lấy quán cà phê
      26	Chọn lấy thẻ cửa hàng tiện lợi
      27	Chọn lấy thẻ rừng
      28	Chọn lấy thẻ nhà máy pho mát
      29	Chọn lấy thẻ nhà máy nội thất
      30	Chọn lấy thẻ mỏ quặng
      31	Chọn lấy thẻ quán ăn gia đình
      32	Chọn lấy thẻ vườn táo
      33	Chọn lấy thẻ chợ trái cây
      34	Mua thẻ lúa mì
      35	Mua thẻ nông trại
      36	Mua thẻ tiệm bánh
      37	Mua thẻ quán cà phê
      38	Mua thẻ cửa hàng tiện lợi
      39	Mua thẻ rừng
      40	Mua thẻ nhà máy pho mát
      41	Mua thẻ nhà máy nội thất
      42	Mua thẻ mỏ quặng
      43	Mua thẻ quán ăn gia đình
      44	Mua thẻ vườn táo
      45	Mua thẻ chợ trái cây
      46	Mua thẻ sân vận động
      47	Mua thẻ đài truyền hình
      48	Mua thẻ trung tâm thương mại
      49	Mua thẻ 22đ
      50	Mua thẻ 16đ
      51	Mua thẻ 10đ
      52	Mua thẻ 4đ
      53	Ko mua thẻ nữa

##  :bust_in_silhouette: P_state
      Quy ước phase: 
      phase1: chọn xúc sắc để đổ
      phase2: chọn đổ lại hay k
      phase3: chọn lấy tiền của ai
      phase4: chọn người để đổi
      phase5: chọn lá bài để đổi
      phase6: chọn lá bài muốn lấy
      phase7: chọn mua thẻ

      Thứ tự thẻ:
            thẻ lúa mì
            thẻ nông trại
            thẻ tiệm bánh
            thẻ quán cà phê
            thẻ cửa hàng tiện lợi
            thẻ rừng
            thẻ nhà máy pho mát
            thẻ nhà máy nội thất
            thẻ mỏ quặng
            thẻ quán ăn gia đình
            thẻ vườn táo
            thẻ chợ trái cây
            thẻ sân vận động
            thẻ đài truyền hình
            thẻ trung tâm thương mại
            thẻ 22đ
            thẻ 16đ
            thẻ 10đ
            thẻ 4đ

      *   [0:20]: Coin và số lượng từng loại thẻ của người chơi
      *   [20:80] Thông tin 3 người chơi còn lại
      *   [80:92]:   số lượng các loại thẻ còn trên bàn
      *   [92:104]:   số lượng các loại thẻ người chơi mua trong turn
      *   [104:116]:   Thẻ người chơi dùng để đổi
      *   [116]:  người chơi được đổ xúc sắc tiếp không
      *   [117]:  giá trị xúc sắc gần nhất
      *   [118:122] : người chơi bị chọn
      *   [122:129]:  Phase game
      *   [129]:  Check end game


##  :globe_with_meridians: ENV_state
      *   [0:80] Thông tin các người chơi (coin, số lượng thẻ mỗi loại): 
      *   [80:92] Số lượng của 12 thẻ bình thường còn trên bàn
      *   [92:104] Số lượng loại thẻ mà người chơi đã mua trong turn
      *   [104:111] Các thông tin khác [card_sell, được đi tiếp hay ko, last_dice, pick_person, id_action, phase, check_end_game]


