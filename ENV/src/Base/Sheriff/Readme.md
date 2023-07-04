Đã dùng 83% bộ nhớ … Bạn có thể giải phóng bộ nhớ hoặc mua thêm bộ nhớ cho Drive, Gmail và Google Photos.
## :dart: Báo cáo MachiKoro
1.   `Tốc độ chạy`
      - **1000 Game**: 
      - **1000 Game full numba**: 
      - **10000 Game**: 

2. `Chuẩn form`: **Đã test**
3. `Đúng luật`: **Đã check**
4. `Không bị loop vô hạn`: **Đã test** với 1000000 ván
5. `Tốc độ chạy các hàm con mà người chơi dùng`: 1000game: 
6. `Số ván check_vic > victory_thật`: chạy 10000 ván thì check_victory = check_winner = 
7. `Giá trị state, action`:
9. `Tối thiểu số lần truyền vào player`: 

## :globe_with_meridians: ENV_state
*   [0:400] **Thông tin các người chơi**: (coin, debt, is_police, typy_bag(4),    coin_bribe, number_smuggle_card, number_bribe_card_in_bag) => 10; 15*6: thẻ hối lộ done, thẻ done, thẻ hối lộ trong túi, thẻ trong túi, thẻ bỏ đi, thẻ trên tay => 90 => tổng mỗi người chơi 100 thông tin/người
*   [400:586] **thẻ bài ở chồng bài úp, có giá trị từ 0 - 15, trong đó 0 đại điện cho ko có thẻ, còn các thẻ còn lại có giá trị như dưới**
NORMAL_CARD = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                        7, 8, 8, 8, 8, 8])

ROYAL_CARD = np.array([ 9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15])

*   [586:711] **thẻ bài ở chồng bài lật bên trái, có giá trị từ 0 - 15, trong đó 0 đại điện cho ko có thẻ**
*   [711:836] **thẻ bài ở chồng bài lật bên phải, có giá trị từ 0 - 15, trong đó 0 đại điện cho ko có thẻ**

*   [836:896] **thẻ mà người chơi bỏ đi trong lượt ( được đặt ở trong chồng bài lật trái và lật phải)**
*   [89:900] **4 vị trí, thể hiện người chơi nào đang bị kiểm tra**
*   [900] **id người chơi đang action**
*   [901] **số lần đổi cảnh sát trưởng**
*   [902] **số người đã bị sherrif kiểm tra trong lượt**
*   [903] **check end**
*   [904] **phase game**





**Total env_state length: 905**
## :bust_in_silhouette: P_state
*   [0:100]: **Thông tin của người chơi chính**:
        0: player coin  : số coin đang có
        1: player debt  : số coin người chơi nợ tổng các ng chơi khác
        2: is_police    : có phải sheriff hay ko
        [3 : 7]: type_in_bag    : loại tài nguyên được khai báo trong túi khi đi chợ
        7: coin_bride           : số coin người chơi hối lộ sheriff
        8: number_smuggle       : số thẻ buôn lậu thành công của người chơi
        9: number_card_bride_bag:  số thẻ trong túi đi chợ người chơi mang ra hối lộ
        -90:-75: card_done_bride: thẻ người chơi đã buôn thành công đem ra hối lộ
        -75:60: card_done       : thẻ người chơi đã buôn thành công
        -60:-45: card_in_bag_bride: thẻ trong túi đi chợ người chơi dùng để hối lộ
        -45:-30: card_in_bag    : thẻ trong túi đi chợ của người chơi
        -30:-15: card_bag_drop  : thẻ trong túi của người chơi bị sheriff tịch thu và đem bỏ
        -15: card_hand          : thẻ trên tay người chơi
                        
*   [100:187] **Thông tin 3 người chơi còn lại**
    #(coin,debt, is_police, typy_bag(4), coin_bribe, number_smuggle_card, number_bribe_card_in_bag) => 10 vị trí
    #thẻ done hối lộ: 15 vị trí, thẻ done chính ngạch của người chơi: 4 vị trí => 19 vị trí
    => 29*3 = 87 vị trí

*   [187:277]:   **Thông tin 6 thẻ đầu tiên ở chồng bài lật trái, cứ 15 vị trí sẽ có 1 vị trí là 1 nếu còn thẻ, nếu ko có thẻ tương ứng thì cả 15 giá trị là 0**:
ví dụ: chồng bài lật còn 2 thẻ thì chỉ có 2 vị trí là 1 lần lượt ở đoạn 183-198 và 198-213
*   [277:367]:   **Thông tin 6 thẻ đầu tiên ở chồng bài lật phải, cứ 15 vị trí sẽ có 1 vị trí là 1 nếu còn thẻ, nếu ko có thẻ tương ứng thì cả 15 giá trị là 0**:

*   [367:412]:   **số lượng các loại thẻ mà các người chơi còn lại đã bỏ ra trong lượt các thẻ bị bỏ vào chồng bài lật**:
*   [412]:  **số lần đổi cảnh sát trưởng**
*   [413]:  **check end**

      

*   [414:425]:   **mảng biểu diễn phase của game**:
*   [425:470]:   **Các thẻ người chơi khác buôn thành công trong game**: Chỉ khác 0 khi game kết thúc, còn lại luôn bằng 0
*   [470:474]: thứ tự hành động của các người chơi
*   [474:478]: số lượng thẻ trong túi của các người chơi


**Total: player state length = 478 **

## :video_game: ALL_ACTION
Action	Mean
0	bỏ thẻ apple
1	bỏ thẻ cheese
2	bỏ thẻ bread
3	bỏ thẻ chicken
4	bỏ thẻ peper
5	bỏ thẻ mead
6	bỏ thẻ silk
7	bỏ thẻ crossbow
8	bỏ thẻ green_apple
9	bỏ thẻ gouda_cheese
10	bỏ thẻ rye_bread
11	bỏ thẻ royal_rooster
12	bỏ thẻ golden_apple
13	bỏ thẻ bleu_cheese
14	bỏ thẻ pump_bread
15	Không bỏ thẻ nữa
16	Lấy thẻ chồng bài rút
17	Lấy thẻ chồng bài lật trái
18	Lấy thẻ chồng bài lật phải
19	Trả thẻ vào chồng bài lật trái
20	Trả thẻ vào chồng bài lật phải
21	bỏ thẻ apple vào túi
22	bỏ thẻ cheese vào túi
23	bỏ thẻ bread vào túi
24	bỏ thẻ chicken vào túi
25	bỏ thẻ peper vào túi
26	bỏ thẻ mead vào túi
27	bỏ thẻ silk vào túi
28	bỏ thẻ crossbow vào túi
29	bỏ thẻ green_apple vào túi
30	bỏ thẻ gouda_cheese vào túi
31	bỏ thẻ rye_bread vào túi
32	bỏ thẻ royal_rooster vào túi
33	bỏ thẻ golden_apple vào túi
34	bỏ thẻ bleu_cheese vào túi
35	bỏ thẻ pump_bread vào túi
36	Không bỏ thẻ vào túi nữa
37	Khai báo hàng là apple
38	Khai báo hàng là cheese
39	Khai báo hàng là bread
40	Khai báo hàng là chicken
41	Kiểm tra người đầu tiên cạnh mình
42	Kiểm tra người thứ 2 cạnh mình
43	Kiểm tra người thứ 3 cạnh mình
44	Không hối lộ coin nữa
45	Hối lộ thêm 1 coin
46	hối lộ thẻ apple done
47	hối lộ thẻ cheese done
48	hối lộ thẻ bread done
49	hối lộ thẻ chicken done
50	hối lộ thẻ peper done
51	hối lộ thẻ mead done
52	hối lộ thẻ silk done
53	hối lộ thẻ crossbow done
54	hối lộ thẻ green_apple done
55	hối lộ thẻ gouda_cheese done
56	hối lộ thẻ rye_bread done
57	hối lộ thẻ royal_rooster done
58	hối lộ thẻ golden_apple done
59	hối lộ thẻ bleu_cheese done
60	hối lộ thẻ pump_bread done
61	Không hối lộ thẻ done nữa
62	hối lộ thẻ apple trong túi
63	hối lộ thẻ cheese trong túi
64	hối lộ thẻ bread trong túi
65	hối lộ thẻ chicken trong túi
66	hối lộ thẻ peper trong túi
67	hối lộ thẻ mead trong túi
68	hối lộ thẻ silk trong túi
69	hối lộ thẻ crossbow trong túi
70	hối lộ thẻ green_apple trong túi
71	hối lộ thẻ gouda_cheese trong túi
72	hối lộ thẻ rye_bread trong túi
73	hối lộ thẻ royal_rooster trong túi
74	hối lộ thẻ golden_apple trong túi
75	hối lộ thẻ bleu_cheese trong túi
76	hối lộ thẻ pump_bread trong túi
77	Không hối lộ thẻ trong túi nữa
78	Kiểm tra hàng
79	Cho qua
80	Bỏ thẻ tịch thu vào bên trái
81	Bỏ thẻ tịch thu vào bên phải

**Total: number all action = 82 **

## :video_game :điều chỉnh so với phiên bản gốc:
* người chơi không thể hối lộ để sheriff kiểm tra hàng hóa người chơi khác trong giai đoạn sheriff kiểm tra người chơi khác
* người chơi không thể hối lộ bằng các ước hẹn trong tương lai, chỉ có thể hối lộ bằng tiền (nếu coin - debt > 0), hối lộ bằng thẻ trong túi đi chợ, hối lộ bằng thẻ đã đi chợ thành công
* khi người chơi không đủ tiền nộp phạt, thay vì phải nộp các thẻ đã đi chợ thành công cho sheriff để trừ nợ, người chơi sẽ bị trừ sạch tiền và khoản tiền thiếu sẽ trở thành nợ. Sheriff sẽ nhận đủ khoản tiền phạt. 
