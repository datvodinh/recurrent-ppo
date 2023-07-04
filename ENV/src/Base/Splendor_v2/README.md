##   Thông tin quan trọng:
     -    17: Điểm của bản thân
     -    213, 214, 215: Điểm của 3 người chơi còn lại
     
##  :video_game: Action
    -  [0]   : Là action bỏ lượt
    -  [1:13]  lấy 12 thẻ trên bàn
    -  [13:16] Là mở 3 thẻ đang úp
    -  [16:28] Úp 12 thẻ trên bàn
    -  [28:31] Úp 3 thẻ ẩn
    -  [31:36] Lấy 5 nguyên liêu
    -  [36:42] Trả 6 nguyên liệu

##  :bust_in_silhouette: P_state
     - Thông tin nguyên liệu theo thứ tự: red, blue, green, black, white, yellow
     -   [:6]            là Số lượng nguyên liệu đang có trên bàn
     -   [6: 18]         thông tin của người chơi, gồm có  6 nguyên liệu đang có, 5 nguyên liệu mặc định và điểm
     -   [18:150]:       12 thẻ bình thường trên bàn, mỗi thẻ có 11 state gồm: [điểm, 5 state loại thẻ mặc định, 5 nguyên liệu cần bỏ ra khi mua thẻ]
               - [18:29]: Thông tin thẻ thứ 0
               - [29:40]: Thông tin thẻ thứ 1
               - [40:51]: Thông tin thẻ thứ 2
               - [51:62]: Thông tin thẻ thứ 3
               - [62:73]: Thông tin thẻ thứ 4
               - [73:84]: Thông tin thẻ thứ 5
               - [84:95]: Thông tin thẻ thứ 6
               - [95:106]: Thông tin thẻ thứ 7
               - [106:117]: Thông tin thẻ thứ 8
               - [117:128]: Thông tin thẻ thứ 9
               - [128:139]: Thông tin thẻ thứ 10
               - [139:150]: Thông tin thẻ thứ 11
     -   [150: 175]:     5 thẻ Noble trên bàn, mỗi thẻ có 5 state gồm: [5 loại nguyên liệu cần]
               - [150:155]: Thông tin thẻ noble thứ 0
               - [155:160]: Thông tin thẻ noble thứ 1
               - [160:165]: Thông tin thẻ noble thứ 2
               - [165:170]: Thông tin thẻ noble thứ 3
               - [170:175]: Thông tin thẻ noble thứ 4
     -   [175:208]:      3 thẻ úp trên tay, mỗi thẻ có 11 state gồm: [điểm, 5 state loại thẻ, 5 nguyên liệu mua] (Thứ tự thẻ theo cấp độ level, có thể xem trong src/Base/Splendor_v2/playing_card_images/Cards)
               - [175:186]: Thông tin thẻ úp thứ 0
               - [186:197]: Thông tin thẻ úp thứ 1
               - [197:208]: Thông tin thẻ úp thứ 2
     -   [208: 213]:     5 nguyên liệu đã lấy trong phase lấy nguyên liệu(Không có ý nghĩa trong game này)
     -   [213:216]:      điểm của 3 người chơi còn lại
     -   [216:219]:      Có thể úp được thẻ ẩn không, (1, 0). Gồm có 3 thẻ ẩn của 3 loại
     -   [219]:          Số thẻ có thể úp trên bàn
     -   [220]:          Đã hết game hay chưa(1, 0)


##  :globe_with_meridians: ENV_state
    -    [0:90]     các thẻ trên bàn: 5 là đang ở trên bàn, -(p_id) là đang úp, p_id là người chơi đã mua được
    -    [100]      Turn
    -    [101:107]  Nguyên liệu trên bàn, gồm có 6 nguyên liệu
    -    [107 + 12 * p_id:119 + 12 * p_id] thông tin của người chơi, gồm có  6 nguyên liệu đang có, 5 nguyên liệu mặc định và điểm
    -    [155:160] 5 Nguyên liệu mà người đó đã lấy trong turn
    -    [161:164] 3 thẻ ẩn có thể úp cấp 1, 2, 3


