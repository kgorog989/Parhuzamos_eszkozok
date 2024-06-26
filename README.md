# Parhuzamos_eszkozok
Párhuzamos eszközök programozása gyakorlat


**Görög Krisztina Erzsébet, MPW46D**

# Féléves feladat

A választott féléves feladatom az ant colony algoritmus (ACO) implementálása a traveling salesman problémára (TSP). Először szekvenciálisan valósítom meg, utána pedig OpenCL-t használva. A két módszert összehasonlítom. 

## ACO

Az ACO (Ant Colony Optimization, hangya kolónia algoritmus) egy olyan algoritmus, amely a hangyák viselkedését veszi alapul egy probléma megoldására. Adott egy kolónia, ami több hangyából áll. A hangyák több iterációban bejárják a gráfként reprezentált problémát, így keresik a megoldást. Az egyes iterációkban feromon utat hagynak maguk után, ami a következő iterációkban történő keresést segíti, hiszen a több feromont tartalmazó utat nagyobb valószínűséggel fogják választani legközelebb. Az algoritmus tartalmaz némi randomizációt, hogy ne mindig ugyanazt az utat járják be a hangyák, így biztosítva a teljes gráf felfedezését. A feromon utak egy idő után az intenzitásukból is veszítenek, ezért az egyes iterációban csökkenn az erősségük is némileg. A feromonoknak köszönhetően juthatunk közelebb az optimális megoldáshoz. Az algoritmus egy részecske-raj optimalizálás algoritmus. 

## TSP

A TSP (Traveling Salesman Problem, utazóügynök probléma) az egyik leginkább kutatott optimalizációs problámák közé tartozik. Legyen adott egy városhalmaz. A probléma arra a kérdésre keresi a választ, hogy hogyan látogathatjuk meg a legrövidebb útvonallal az összes várost úgy, hogy végül a kiindulási városba térünk vissza. A problémát leggyakrabban gráfhalmazzal szokták reprezentálni. 

## Megvalósítás, futási eredmények

A feladatot a *beadando* mappában oldottam meg. Az *aco_opencl* az OpenCL-ben történő, az *aco_seq* a szekvenciális megoldást tartalmazza. Mindegyik programot teszteltem 22 és 312 város esetén is. A városok közötti távolságokat tartalmazó mátrixok a *data* mappában találhatók (22 város esetén: *ger22.txt*, 312 esetén: *usca312.txt*). Ugyanebbe a mappába mentettem az egyes futási eredményeket *times_MÁTRIXNEVE.txt* néven. 


Az OpenCL-ben történő megvalósításhoz bizonyos tömbök megvalósításánál 3D-s tömbök használatára volt szükségem. Ez a kernel lehető leghatékonyabb megvalósítása érdekében történt. A tömbök a következőek: *ant_tours, ant_randoms, visited_cities*. Mindegyiknél az első dimenzió a hangyák száma, a második az iterációk száma, a harmadik a városok száma. Emiatt az *ant_lengths* két dimenziós lett, hiszen benne is fontos az iterációk száma. Első dimenziója a hangyák száma, második az iterációk száma. 


A következő ábrán a futási eredmények grafikus szemléltetése látható.



<img width="100%" alt="image" src="https://github.com/kgorog989/Parhuzamos_eszkozok/assets/90555277/61244e93-95a2-4697-9f52-4ebb2102d041">



A következő alpontokban a futtatások eredményei találhatók. A eredmények alapján nem éri meg az OpenCL használata. A párhuzamosítás ellenére több időt vesz igénybe, mint a szekvenciális módszer használata. Más módszerek használata eredményesebb lehet. A legjobb út keresése viszont mindkét esetben hasonlóan hatékony. 

### 22 város esetén

#### Szekvenciális program

| Hangyák száma | Futási idő | Legjobb út hossza |
|-----|----------|------------|
| 2   | 0.033000 | 871.000000 |
| 3   | 0.023000 | 834.000000 |
| 4   | 0.047000 | 789.000000 |
| 5   | 0.037000 | 789.000000 |
| 6   | 0.041000 | 833.000000 |
| 7   | 0.035000 | 789.000000 |
| 8   | 0.038000 | 813.000000 |
| 9   | 0.044000 | 785.000000 |
| 10  | 0.049000 | 805.000000 |
| 11  | 0.062000 | 801.000000 |
| 12  | 0.062000 | 789.000000 |
| 13  | 0.073000 | 790.000000 |
| 14  | 0.084000 | 789.000000 |
| 15  | 0.069000 | 789.000000 |
| 16  | 0.120000 | 801.000000 |
| 17  | 0.097000 | 790.000000 |
| 18  | 0.129000 | 789.000000 |
| 19  | 0.106000 | 789.000000 |
| 20  | 0.092000 | 789.000000 |
| 21  | 0.125000 | 789.000000 |
| 22  | 0.127000 | 782.000000 |
| 23  | 0.124000 | 797.000000 |
| 24  | 0.141000 | 789.000000 |
| 25  | 0.112000 | 789.000000 |
| 26  | 0.155000 | 805.000000 |
| 27  | 0.167000 | 801.000000 |
| 28  | 0.189000 | 785.000000 |
| 29  | 0.130000 | 786.000000 |
| 30  | 0.135000 | 789.000000 |
| 31  | 0.141000 | 805.000000 |
| 32  | 0.146000 | 786.000000 |
| 33  | 0.148000 | 785.000000 |
| 34  | 0.155000 | 781.000000 |
| 35  | 0.157000 | 789.000000 |
| 36  | 0.165000 | 789.000000 |
| 37  | 0.193000 | 781.000000 |
| 38  | 0.203000 | 785.000000 |
| 39  | 0.213000 | 789.000000 |
| 40  | 0.190000 | 789.000000 |
| 41  | 0.200000 | 782.000000 |
| 42  | 0.199000 | 789.000000 |
| 43  | 0.235000 | 789.000000 |
| 44  | 0.212000 | 789.000000 |
| 45  | 0.215000 | 789.000000 |
| 46  | 0.248000 | 789.000000 |
| 47  | 0.243000 | 789.000000 |
| 48  | 0.256000 | 785.000000 |
| 49  | 0.251000 | 781.000000 |
| 50  | 0.222000 | 781.000000 |
| 51  | 0.224000 | 790.000000 |
| 52  | 0.230000 | 781.000000 |
| 53  | 0.267000 | 785.000000 |
| 54  | 0.268000 | 781.000000 |
| 55  | 0.233000 | 781.000000 |
| 56  | 0.237000 | 789.000000 |
| 57  | 0.241000 | 781.000000 |
| 58  | 0.250000 | 789.000000 |
| 59  | 0.275000 | 789.000000 |
| 60  | 0.262000 | 781.000000 |
| 61  | 0.261000 | 789.000000 |
| 62  | 0.265000 | 785.000000 |
| 63  | 0.346000 | 789.000000 |
| 64  | 0.304000 | 781.000000 |
| 65  | 0.275000 | 785.000000 |
| 66  | 0.315000 | 785.000000 |
| 67  | 0.320000 | 789.000000 |
| 68  | 0.295000 | 785.000000 |
| 69  | 0.331000 | 786.000000 |
| 70  | 0.330000 | 781.000000 |
| 71  | 0.340000 | 781.000000 |
| 72  | 0.379000 | 785.000000 |
| 73  | 0.380000 | 781.000000 |
| 74  | 0.334000 | 789.000000 |
| 75  | 0.355000 | 785.000000 |
| 76  | 0.357000 | 781.000000 |
| 77  | 0.363000 | 785.000000 |
| 78  | 0.370000 | 789.000000 |
| 79  | 0.373000 | 785.000000 |
| 80  | 0.384000 | 781.000000 |
| 81  | 0.375000 | 785.000000 |
| 82  | 0.366000 | 785.000000 |
| 83  | 0.350000 | 785.000000 |
| 84  | 0.354000 | 789.000000 |
| 85  | 0.405000 | 781.000000 |
| 86  | 0.399000 | 781.000000 |
| 87  | 0.427000 | 781.000000 |
| 88  | 0.407000 | 781.000000 |
| 89  | 0.373000 | 781.000000 |
| 90  | 0.380000 | 785.000000 |
| 91  | 0.384000 | 781.000000 |
| 92  | 0.421000 | 781.000000 |
| 93  | 0.391000 | 781.000000 |
| 94  | 0.404000 | 781.000000 |
| 95  | 0.491000 | 781.000000 |
| 96  | 0.404000 | 786.000000 |
| 97  | 0.491000 | 781.000000 |
| 98  | 0.488000 | 781.000000 |
| 99  | 0.454000 | 785.000000 |
| 100 | 0.449000 | 785.000000 |

#### OpenCL program

| Hangyák száma | Teljes futási idő | Kernelben töltött idő | Legjobb út hossza |
|-----|----------|----------|------------|
| 2   | 0.181000 | 0.171271 | 822.000000 |
| 3   | 0.208000 | 0.198000 | 789.000000 |
| 4   | 0.223000 | 0.214165 | 863.000000 |
| 5   | 0.245000 | 0.234209 | 829.000000 |
| 6   | 0.256000 | 0.246337 | 805.000000 |
| 7   | 0.281000 | 0.256035 | 789.000000 |
| 8   | 0.278000 | 0.265121 | 819.000000 |
| 9   | 0.319000 | 0.308416 | 828.000000 |
| 10  | 0.339000 | 0.328898 | 789.000000 |
| 11  | 0.356000 | 0.343893 | 789.000000 |
| 12  | 0.366000 | 0.355099 | 789.000000 |
| 13  | 0.378000 | 0.367398 | 785.000000 |
| 14  | 0.389000 | 0.376322 | 789.000000 |
| 15  | 0.396000 | 0.383360 | 800.000000 |
| 16  | 0.401000 | 0.389037 | 808.000000 |
| 17  | 0.404000 | 0.391330 | 789.000000 |
| 18  | 0.407000 | 0.393093 | 789.000000 |
| 19  | 0.404000 | 0.394992 | 789.000000 |
| 20  | 0.409000 | 0.397439 | 785.000000 |
| 21  | 0.415000 | 0.399709 | 782.000000 |
| 22  | 0.415000 | 0.401252 | 781.000000 |
| 23  | 0.415000 | 0.403186 | 789.000000 |
| 24  | 0.420000 | 0.405416 | 785.000000 |
| 25  | 0.422000 | 0.407923 | 785.000000 |
| 26  | 0.425000 | 0.409556 | 789.000000 |
| 27  | 0.429000 | 0.412085 | 789.000000 |
| 28  | 0.430000 | 0.414096 | 785.000000 |
| 29  | 0.431000 | 0.416520 | 789.000000 |
| 30  | 0.437000 | 0.418595 | 785.000000 |
| 31  | 0.436000 | 0.422111 | 789.000000 |
| 32  | 0.442000 | 0.425376 | 782.000000 |
| 33  | 0.442000 | 0.427786 | 785.000000 |
| 34  | 0.447000 | 0.429898 | 786.000000 |
| 35  | 0.448000 | 0.431217 | 789.000000 |
| 36  | 0.449000 | 0.433898 | 785.000000 |
| 37  | 0.456000 | 0.435136 | 789.000000 |
| 38  | 0.457000 | 0.438617 | 789.000000 |
| 39  | 0.453000 | 0.439466 | 789.000000 |
| 40  | 0.463000 | 0.442177 | 789.000000 |
| 41  | 0.461000 | 0.444381 | 789.000000 |
| 42  | 0.467000 | 0.446492 | 781.000000 |
| 43  | 0.465000 | 0.448887 | 781.000000 |
| 44  | 0.469000 | 0.450636 | 789.000000 |
| 45  | 0.470000 | 0.452945 | 785.000000 |
| 46  | 0.478000 | 0.454612 | 785.000000 |
| 47  | 0.479000 | 0.457181 | 785.000000 |
| 48  | 0.478000 | 0.460246 | 789.000000 |
| 49  | 0.478000 | 0.461946 | 786.000000 |
| 50  | 0.486000 | 0.464827 | 789.000000 |
| 51  | 0.487000 | 0.466495 | 781.000000 |
| 52  | 0.491000 | 0.468663 | 781.000000 |
| 53  | 0.495000 | 0.471121 | 785.000000 |
| 54  | 0.495000 | 0.472583 | 782.000000 |
| 55  | 0.495000 | 0.474781 | 789.000000 |
| 56  | 0.499000 | 0.476877 | 786.000000 |
| 57  | 0.495000 | 0.479161 | 781.000000 |
| 58  | 0.507000 | 0.481361 | 789.000000 |
| 59  | 0.504000 | 0.483110 | 781.000000 |
| 60  | 0.510000 | 0.485968 | 785.000000 |
| 61  | 0.510000 | 0.487466 | 789.000000 |
| 62  | 0.513000 | 0.490115 | 781.000000 |
| 63  | 0.512000 | 0.491124 | 785.000000 |
| 64  | 0.518000 | 0.494499 | 781.000000 |
| 65  | 0.520000 | 0.497017 | 781.000000 |
| 66  | 0.525000 | 0.498843 | 789.000000 |
| 67  | 0.518000 | 0.500525 | 781.000000 |
| 68  | 0.526000 | 0.502466 | 781.000000 |
| 69  | 0.531000 | 0.504744 | 781.000000 |
| 70  | 0.532000 | 0.507201 | 785.000000 |
| 71  | 0.537000 | 0.509066 | 781.000000 |
| 72  | 0.536000 | 0.511582 | 785.000000 |
| 73  | 0.539000 | 0.513240 | 781.000000 |
| 74  | 0.543000 | 0.515399 | 781.000000 |
| 75  | 0.545000 | 0.517699 | 781.000000 |
| 76  | 0.546000 | 0.519608 | 781.000000 |
| 77  | 0.550000 | 0.521729 | 781.000000 |
| 78  | 0.554000 | 0.523966 | 785.000000 |
| 79  | 0.545000 | 0.526324 | 786.000000 |
| 80  | 0.556000 | 0.529213 | 781.000000 |
| 81  | 0.557000 | 0.531679 | 782.000000 |
| 82  | 0.565000 | 0.533504 | 781.000000 |
| 83  | 0.557000 | 0.536189 | 781.000000 |
| 84  | 0.566000 | 0.537713 | 781.000000 |
| 85  | 0.569000 | 0.540138 | 781.000000 |
| 86  | 0.570000 | 0.542081 | 781.000000 |
| 87  | 0.575000 | 0.544483 | 781.000000 |
| 88  | 0.575000 | 0.545811 | 785.000000 |
| 89  | 0.578000 | 0.547877 | 782.000000 |
| 90  | 0.580000 | 0.550090 | 781.000000 |
| 91  | 0.583000 | 0.552134 | 789.000000 |
| 92  | 0.585000 | 0.554169 | 781.000000 |
| 93  | 0.588000 | 0.556602 | 782.000000 |
| 94  | 0.580000 | 0.558761 | 785.000000 |
| 95  | 0.594000 | 0.561283 | 785.000000 |
| 96  | 0.596000 | 0.563829 | 785.000000 |
| 97  | 0.596000 | 0.565630 | 789.000000 |
| 98  | 0.598000 | 0.567609 | 785.000000 |
| 99  | 0.596000 | 0.570119 | 781.000000 |
| 100 | 0.604000 | 0.571997 | 786.000000 |

### 312 város esetén

#### Szekvenciális program

| Hangyák száma | Futási idő | Legjobb út hossza |
|-----|------------|--------------|
| 2   | 1.974000   | 53003.000000 |
| 3   | 2.940000   | 52000.000000 |
| 4   | 3.649000   | 49101.000000 |
| 5   | 4.245000   | 52543.000000 |
| 6   | 7.486000   | 47919.000000 |
| 7   | 7.781000   | 48743.000000 |
| 8   | 6.773000   | 47015.000000 |
| 9   | 7.400000   | 49354.000000 |
| 10  | 8.254000   | 47885.000000 |
| 11  | 9.041000   | 47006.000000 |
| 12  | 9.712000   | 48197.000000 |
| 13  | 10.627000  | 47465.000000 |
| 14  | 12.278000  | 47018.000000 |
| 15  | 12.226000  | 46542.000000 |
| 16  | 13.118000  | 48422.000000 |
| 17  | 13.865000  | 48003.000000 |
| 18  | 14.794000  | 47623.000000 |
| 19  | 15.531000  | 46372.000000 |
| 20  | 16.212000  | 47286.000000 |
| 21  | 17.169000  | 46786.000000 |
| 22  | 18.098000  | 47357.000000 |
| 23  | 18.534000  | 46963.000000 |
| 24  | 19.654000  | 46835.000000 |
| 25  | 20.250000  | 46768.000000 |
| 26  | 21.182000  | 46853.000000 |
| 27  | 21.850000  | 46983.000000 |
| 28  | 22.695000  | 45820.000000 |
| 29  | 23.519000  | 45968.000000 |
| 30  | 24.437000  | 46119.000000 |
| 31  | 25.168000  | 45312.000000 |
| 32  | 25.822000  | 47172.000000 |
| 33  | 27.176000  | 45984.000000 |
| 34  | 27.495000  | 46851.000000 |
| 35  | 28.269000  | 46897.000000 |
| 36  | 29.070000  | 44304.000000 |
| 37  | 30.118000  | 46219.000000 |
| 38  | 30.886000  | 45083.000000 |
| 39  | 31.498000  | 46113.000000 |
| 40  | 32.602000  | 46102.000000 |
| 41  | 33.301000  | 46018.000000 |
| 42  | 33.983000  | 46478.000000 |
| 43  | 34.835000  | 46279.000000 |
| 44  | 35.650000  | 45257.000000 |
| 45  | 36.435000  | 46216.000000 |
| 46  | 37.124000  | 46005.000000 |
| 47  | 38.292000  | 46307.000000 |
| 48  | 38.859000  | 46238.000000 |
| 49  | 39.740000  | 44904.000000 |
| 50  | 40.285000  | 45858.000000 |
| 51  | 41.123000  | 45438.000000 |
| 52  | 42.015000  | 46541.000000 |
| 53  | 43.041000  | 45518.000000 |
| 54  | 43.642000  | 46005.000000 |
| 55  | 44.763000  | 46001.000000 |
| 56  | 45.203000  | 45886.000000 |
| 57  | 46.246000  | 46105.000000 |
| 58  | 46.959000  | 46050.000000 |
| 59  | 48.066000  | 45729.000000 |
| 60  | 48.497000  | 44376.000000 |
| 61  | 49.254000  | 45216.000000 |
| 62  | 50.212000  | 45413.000000 |
| 63  | 51.011000  | 44889.000000 |
| 64  | 51.659000  | 45387.000000 |
| 65  | 52.667000  | 46345.000000 |
| 66  | 53.494000  | 46345.000000 |
| 67  | 54.202000  | 45661.000000 |
| 68  | 54.918000  | 44633.000000 |
| 69  | 55.840000  | 44733.000000 |
| 70  | 56.398000  | 44552.000000 |
| 71  | 57.340000  | 45424.000000 |
| 72  | 57.880000  | 44966.000000 |
| 73  | 59.540000  | 45359.000000 |
| 74  | 59.752000  | 45165.000000 |
| 75  | 83.008000  | 45332.000000 |
| 76  | 84.037000  | 45428.000000 |
| 77  | 77.258000  | 45461.000000 |
| 78  | 73.288000  | 44755.000000 |
| 79  | 63.904000  | 44480.000000 |
| 80  | 64.705000  | 46121.000000 |
| 81  | 68.073000  | 44969.000000 |
| 82  | 68.083000  | 45738.000000 |
| 83  | 67.207000  | 44159.000000 |
| 84  | 69.688000  | 45275.000000 |
| 85  | 80.752000  | 44336.000000 |
| 86  | 76.237000  | 45439.000000 |
| 87  | 111.582000 | 44697.000000 |
| 88  | 113.398000 | 44806.000000 |
| 89  | 112.329000 | 45517.000000 |
| 90  | 112.728000 | 44949.000000 |
| 91  | 91.492000  | 45048.000000 |
| 92  | 75.178000  | 46056.000000 |
| 93  | 89.615000  | 45164.000000 |
| 94  | 75.787000  | 45099.000000 |
| 95  | 76.947000  | 45734.000000 |
| 96  | 91.046000  | 43626.000000 |
| 97  | 78.584000  | 45438.000000 |
| 98  | 79.445000  | 45239.000000 |
| 99  | 80.847000  | 45233.000000 |
| 100 | 81.809000  | 45264.000000 |

#### OpenCL program

A túl hosszú futási idők miatt futásonként tízzel növeltem a hangyák számát. Az eredmény így is szemléletes. 

| Hangyák száma | Teljes futási idő | Kernelben töltött idő | Legjobb út hossza |
|-----|------------|------------|--------------|
| 10  | 37.923000  | 37.924605  | 48583.000000 |
| 20  | 41.874000  | 41.848372  | 47702.000000 |
| 30  | 51.912000  | 51.903039  | 46153.000000 |
| 40  | 52.040000  | 52.014088  | 47510.000000 |
| 50  | 67.104000  | 67.067906  | 46069.000000 |
| 60  | 84.906000  | 84.870011  | 48478.000000 |
| 70  | 84.910000  | 84.868587  | 48999.000000 |
| 80  | 100.184000 | 100.135389 | 45790.000000 |
| 90  | 116.718000 | 116.658317 | 45863.000000 |
| 100 | 133.190000 | 133.117251 | 44293.000000 |
