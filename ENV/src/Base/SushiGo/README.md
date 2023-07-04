##   Thông tin quan trọng:
        14: Điểm của bản thân
        28, 42, 56, 70: Điểm của những người chơi khác

##  :video_game: Action
        0 | Tempura      
        1 | Sashimi      
        2 | Dumpling     
        3 | 1 Maki Roll  
        4 | 2 Maki Roll  
        5 | 3 Maki Roll  
        6 | Salmon Nigiri
        7 | Squid Nigiri 
        8 | Egg Nigiri   
        9 | Pudding      
        10| Wasabi 
        11| Chopsticks
        12| Use Chopsticks
        13| End Turn

##  :bust_in_silhouette: P_state
        Thứ tự thẻ:         
                0 | Tempura      
                1 | Sashimi      
                2 | Dumpling     
                3 | 1 Maki Roll  
                4 | 2 Maki Roll  
                5 | 3 Maki Roll  
                6 | Salmon Nigiri
                7 | Squid Nigiri 
                8 | Egg Nigiri   
                9 | Pudding      
                10| Wasabi 
                11| Chopsticks
        [0:1]: Vòng chơi
        [1:2]: Lượt chơi
        [2:14]: Số lượng thẻ để người chơi lựa chọn(Gồm 12 thẻ)
        [14*i: 14*(i + 1)]: với i = 0, 1, 2, 3, 4
                + i = 0 là bản thân mình, còn lại là người khác
                + 0: điểm
                + 1: Số lượng thẻ puding
                + 2 -> 14 : Số lượng thẻ trên tay của người chơi
        
        [-2]: Stack dùng thẻ Chopsticks
        [-1]: Stack sử dụng bình thường


#  Env_State
        [0:3]: Vòng chơi, Lượt chơi, Số người chơi(mặc định 5)
        [3:108]: Thẻ trong chồng bài
        [index_2:index_2+9]: (index_2 = 108 + 108+i*9)
            + index_2: index_2+2: điểm và số lượng thẻ puding
            + index_2+2 : index_2+9: Thẻ trên tay của người chơi
