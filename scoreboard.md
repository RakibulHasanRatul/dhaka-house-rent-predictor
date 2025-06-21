# Scoreboard for Locations

While the program results refined for some locations, it is not as accurate as it should be. So, It will be vital to score each locations based on their performance metrics ($R^2$, MSE, MAE). A score **out of 10** is given to each locations based on their performance metrics.

## Scoring Criteria

10 times of the $R^2$ value can be used as the scoring system.

Why? $R^2$ value ranges up to 1, and 10 times of it ranges up to 10. Certainly, as the $R^2$ value increases, the MSE and MAE (error values) also decreases.

However, for ensuring fair test, I provided two scoreboards. [**Scoreboard 01**](#scoreboard-01) is by using the `modified_preprocess_loaded_data` function defined in [modified_preprocessing.py](./modified_preprocessing.py). The script is in the [01_generate_scoreboard.py](./scoreboard_scripts/01_generate_scoreboard.py) file.

And, [**Scoreboard 02**](#scoreboard-02) represents the original `construct_location_from_area` function defined in [app/handler/data/preprocess.py](./app/handler/data/preprocess.py). The script is in the [02_generate_scoreboard.py](./scoreboard_scripts/02_generate_scoreboard.py) file.

## Scoreboard 01

| Location               | Score   |
| ---------------------- | ------- |
| Nikunja, Dhaka         | 9.97    |
| Turag, Dhaka           | 9.25    |
| Shyamoli, Dhaka        | 8.9     |
| Malibagh, Dhaka        | 8.74    |
| Maghbazar, Dhaka       | 8.03    |
| Khilkhet, Dhaka        | 8.02    |
| Bashundhara R-A, Dhaka | 8.01    |
| Badda, Dhaka           | 7.91    |
| New Market, Dhaka      | 7.82    |
| Baridhara, Dhaka       | 7.74    |
| Uttara, Dhaka          | 7.72    |
| Motijheel, Dhaka       | 7.56    |
| Lalbagh, Dhaka         | 7.51    |
| Khilgaon, Dhaka        | 6.7     |
| Cantonment, Dhaka      | 6.68    |
| Banasree, Dhaka        | 6.39    |
| Mirpur, Dhaka          | 5.92    |
| Banani, Dhaka          | 5.7     |
| Dakshin Khan, Dhaka    | 5.52    |
| Rampura, Dhaka         | 5.48    |
| Adabor, Dhaka          | 5.46    |
| Shegunbagicha, Dhaka   | 5.44    |
| Hazaribag, Dhaka       | 5.37    |
| Dhanmondi, Dhaka       | 4.95    |
| Baridhara Dohs, Dhaka  | 4.81    |
| Tejgaon, Dhaka         | 4.5     |
| Shiddheswari, Dhaka    | 4.34    |
| Gulshan, Dhaka         | 4.07    |
| Mohammadpur, Dhaka     | 3.84    |
| Aftab Nagar, Dhaka     | 3.82    |
| Eskaton, Dhaka         | 3.47    |
| Ibrahimpur, Dhaka      | 1.95    |
| Hatirpool, Dhaka       | 0.0     |
| Joar Sahara, Dhaka     | 0.0     |
| Kalachandpur, Dhaka    | 0.0     |
| Kathalbagan, Dhaka     | 0.0     |
| Lalmatia, Dhaka        | 0.0     |
| Niketan, Dhaka         | 0.0     |
| Shahjahanpur, Dhaka    | 0.0     |
| Taltola, Dhaka         | 0.0     |
| Agargaon, Dhaka        | -0.06   |
| Uttar Khan, Dhaka      | -0.37   |
| Shantinagar, Dhaka     | -3.0    |
| Mohakhali Dohs, Dhaka  | -5.17   |
| Bashabo, Dhaka         | -5.51   |
| Kalabagan, Dhaka       | -7.53   |
| Sutrapur, Dhaka        | -68.38  |
| Banani Dohs, Dhaka     | -112.49 |

> Locations having insufficient datasets!

| Location                  | Score |
| ------------------------- | ----- |
| Bangshal, Dhaka           | N/A   |
| Demra, Dhaka              | N/A   |
| Jatra Bari, Dhaka         | N/A   |
| Kachukhet, Dhaka          | N/A   |
| Kafrul, Dhaka             | N/A   |
| Kuril, Dhaka              | N/A   |
| Mohakhali, Dhaka          | N/A   |
| North Shahjahanpur, Dhaka | N/A   |
| Paribagh, Dhaka           | N/A   |
| Pink City, Dhaka          | N/A   |
| Shahbagh, Dhaka           | N/A   |
| Zafrabad, Dhaka           | N/A   |

## Scoreboard 02

| Location                                       | Score       |
| ---------------------------------------------- | ----------- |
| Turag, Dhaka                                   | 9.96        |
| New Market, Dhaka                              | 9.89        |
| Khilkhet, Dhaka                                | 9.83        |
| Agargaon, Dhaka                                | 9.72        |
| Block E, Banasree, Dhaka                       | 9.6         |
| West Rampura, Rampura, Dhaka                   | 9.49        |
| Sector 11, Uttara, Dhaka                       | 9.37        |
| Block E, Aftab Nagar, Dhaka                    | 9.24        |
| Bot Tola, Khilkhet, Dhaka                      | 9.21        |
| Ashkona, Dakshin Khan, Dhaka                   | 8.95        |
| Section 2, Mirpur, Dhaka                       | 8.94        |
| Shyamoli, Dhaka                                | 8.89        |
| Block H, Bashundhara R-A, Dhaka                | 8.73        |
| Ibrahimpur, Dhaka                              | 8.68        |
| Sector 3, Uttara, Dhaka                        | 8.68        |
| Manikdi, Cantonment, Dhaka                     | 8.54        |
| Cantonment, Dhaka                              | 8.51        |
| Block C, Bashundhara R-A, Dhaka                | 8.44        |
| Gawair, Dakshin Khan, Dhaka                    | 8.39        |
| Kallyanpur, Mirpur, Dhaka                      | 8.34        |
| Block F, Bashundhara R-A, Dhaka                | 8.29        |
| Middle Paikpara, Mirpur, Dhaka                 | 8.2         |
| Block I, Bashundhara R-A, Dhaka                | 8.17        |
| Mirpur, Dhaka                                  | 8.03        |
| Maghbazar, Dhaka                               | 7.97        |
| Mohammadpur, Dhaka                             | 7.97        |
| Sector 13, Uttara, Dhaka                       | 7.88        |
| Solmaid, Badda, Dhaka                          | 7.77        |
| Paikpara, Mirpur, Dhaka                        | 7.32        |
| Baridhara, Dhaka                               | 7.21        |
| Taltola, Khilgaon, Dhaka                       | 7.14        |
| Ahmed Nagar, Mirpur, Dhaka                     | 7.04        |
| Sector 4, Uttara, Dhaka                        | 7.03        |
| Gulshan 2, Gulshan, Dhaka                      | 6.86        |
| Section 11, Mirpur, Dhaka                      | 6.81        |
| South Banasree Project, Banasree, Dhaka        | 6.79        |
| Block K, Baridhara, Dhaka                      | 6.68        |
| Section 12, Mirpur, Dhaka                      | 6.66        |
| Sector 10, Uttara, Dhaka                       | 6.53        |
| Madhya Ajampur, Dakshin Khan, Dhaka            | 6.4         |
| Uttar Badda, Badda, Dhaka                      | 6.34        |
| Section 10, Mirpur, Dhaka                      | 5.88        |
| Banani, Dhaka                                  | 5.7         |
| Sector 5, Uttara, Dhaka                        | 5.69        |
| Section 1, Mirpur, Dhaka                       | 5.63        |
| West Dhanmondi And Shangkar, Dhanmondi, Dhaka  | 5.47        |
| Shegunbagicha, Dhaka                           | 5.44        |
| Baitul Aman Housing Society, Adabor, Dhaka     | 5.12        |
| Pc Culture Housing, Mohammadpur, Dhaka         | 5.09        |
| Baridhara Dohs, Dhaka                          | 4.81        |
| Block D, Bashundhara R-A, Dhaka                | 4.74        |
| Mansurabad Housing Society, Adabor, Dhaka      | 4.71        |
| Sector 7, Uttara, Dhaka                        | 4.71        |
| Eskaton, Dhaka                                 | 4.56        |
| Shiddheswari, Dhaka                            | 4.34        |
| West Shewrapara, Mirpur, Dhaka                 | 4.32        |
| East Rampura, Rampura, Dhaka                   | 3.74        |
| Nobodoy Housing Society, Mohammadpur, Dhaka    | 3.71        |
| West Agargaon, Agargaon, Dhaka                 | 3.5         |
| Sector 6, Uttara, Dhaka                        | 3.38        |
| Block H, Banasree, Dhaka                       | 2.92        |
| Sector 9, Uttara, Dhaka                        | 2.68        |
| Block D, Banasree, Dhaka                       | 2.28        |
| Gulshan 1, Gulshan, Dhaka                      | 1.74        |
| Sector 15, Uttara, Dhaka                       | 0.36        |
| Khilgaon, Dhaka                                | 0.24        |
| 2Nd Colony, Mirpur, Dhaka                      | 0.0         |
| Adabor, Dhaka                                  | 0.0         |
| Baigertek, Cantonment, Dhaka                   | 0.0         |
| Bashabo, Dhaka                                 | 0.0         |
| Bhagalpur, Hazaribag, Dhaka                    | 0.0         |
| Block A, Banasree, Dhaka                       | 0.0         |
| Block C, Aftab Nagar, Dhaka                    | 0.0         |
| Block C, Banasree, Dhaka                       | 0.0         |
| Block D, Aftab Nagar, Dhaka                    | 0.0         |
| Block G, Banasree, Dhaka                       | 0.0         |
| Chad Uddan Housing, Mohammadpur, Dhaka         | 0.0         |
| Chandrima Model Town, Mohammadpur, Dhaka       | 0.0         |
| Dhorangartek, Turag, Dhaka                     | 0.0         |
| East Bashabo, Bashabo, Dhaka                   | 0.0         |
| East Mollartek, Dakshin Khan, Dhaka            | 0.0         |
| East Monipur, Mirpur, Dhaka                    | 0.0         |
| East Shewrapara, Mirpur, Dhaka                 | 0.0         |
| Gandaria, Sutrapur, Dhaka                      | 0.0         |
| Gulbag, Malibagh, Dhaka                        | 0.0         |
| Hatirpool, Dhaka                               | 0.0         |
| Hazaribag, Dhaka                               | 0.0         |
| Janata Housing Society, Adabor, Dhaka          | 0.0         |
| Jigatola, Hazaribag, Dhaka                     | 0.0         |
| Joar Sahara, Dhaka                             | 0.0         |
| Kalachandpur, Dhaka                            | 0.0         |
| Kathalbagan, Dhaka                             | 0.0         |
| Kazibari, Dakshin Khan, Dhaka                  | 0.0         |
| Khaje Dewan, Lalbagh, Dhaka                    | 0.0         |
| Lalbagh Road, Lalbagh, Dhaka                   | 0.0         |
| Lalbagh, Dhaka                                 | 0.0         |
| Malibagh, Dhaka                                | 0.0         |
| Matikata, Cantonment, Dhaka                    | 0.0         |
| Merul Badda, Badda, Dhaka                      | 0.0         |
| Mohammadi Housing Ltd., Mohammadpur, Dhaka     | 0.0         |
| Mohammadia Housing Society, Mohammadpur, Dhaka | 0.0         |
| Motijheel, Dhaka                               | 0.0         |
| Naya Paltan, Motijheel, Dhaka                  | 0.0         |
| Nayatola, Maghbazar, Dhaka                     | 0.0         |
| North Ibrahimpur, Ibrahimpur, Dhaka            | 0.0         |
| Nurer Chala, Badda, Dhaka                      | 0.0         |
| Pirerbag, Mirpur, Dhaka                        | 0.0         |
| Sat Masjid Housing, Mohammadpur, Dhaka         | 0.0         |
| Section 13, Mirpur, Dhaka                      | 0.0         |
| Section 3, Mirpur, Dhaka                       | 0.0         |
| Shahjahanpur, Dhaka                            | 0.0         |
| Shyamoli Housing (2Nd Project), Adabor, Dhaka  | 0.0         |
| South Azampur, Dakshin Khan, Dhaka             | 0.0         |
| South Mollartek, Dakshin Khan, Dhaka           | 0.0         |
| South Monipur, Mirpur, Dhaka                   | 0.0         |
| Tajmahal Road, Mohammadpur, Dhaka              | 0.0         |
| Talertek, Khilkhet, Dhaka                      | 0.0         |
| Taltola, Agargaon, Dhaka                       | 0.0         |
| Tejgaon, Dhaka                                 | 0.0         |
| Tilpapara, Khilgaon, Dhaka                     | 0.0         |
| Faydabad, Dakshin Khan, Dhaka                  | -0.06       |
| Aainusbag, Dakshin Khan, Dhaka                 | -0.68       |
| Dhanmondi, Dhaka                               | -0.98       |
| Kawlar, Dakshin Khan, Dhaka                    | -0.98       |
| Badda, Dhaka                                   | -1.06       |
| Dakkhin Paikpara, Mirpur, Dhaka                | -1.66       |
| Sector 1, Uttara, Dhaka                        | -1.79       |
| Block A, Bashundhara R-A, Dhaka                | -2.86       |
| Shantinagar, Dhaka                             | -3.0        |
| Dolipara, Uttara, Dhaka                        | -3.32       |
| East Kazipara, Mirpur, Dhaka                   | -3.52       |
| Mohakhali Dohs, Dhaka                          | -5.17       |
| Mirpur Dohs, Mirpur, Dhaka                     | -6.67       |
| Kalabagan, Dhaka                               | -7.53       |
| South Badda, Badda, Dhaka                      | -8.01       |
| Middle Badda, Badda, Dhaka                     | -8.84       |
| Katashur, Mohammadpur, Dhaka                   | -12.73      |
| Uttar Khan, Dhaka                              | -15.05      |
| Block L, Bashundhara R-A, Dhaka                | -16.46      |
| Sector 14, Uttara, Dhaka                       | -16.56      |
| Shekhertek, Mohammadpur, Dhaka                 | -16.92      |
| Bashundhara R-A, Dhaka                         | -18.53      |
| Pallabi, Mirpur, Dhaka                         | -21.9       |
| Bosila, Mohammadpur, Dhaka                     | -22.14      |
| Rajabazar, Tejgaon, Dhaka                      | -22.54      |
| Dakshin Khan, Dhaka                            | -26.61      |
| Adarsha Para, Uttar Khan, Dhaka                | -28.57      |
| Block J, Bashundhara R-A, Dhaka                | -33.3       |
| Vatara, Badda, Dhaka                           | -36.29      |
| Block K, Bashundhara R-A, Dhaka                | -36.33      |
| Sector 12, Uttara, Dhaka                       | -43.73      |
| Nikunja 2, Nikunja, Dhaka                      | -46.27      |
| Aziz Moholla, Mohammadpur, Dhaka               | -46.82      |
| West Kazipara, Mirpur, Dhaka                   | -52.46      |
| Block A, Aftab Nagar, Dhaka                    | -52.76      |
| East Azampur, Dakshin Khan, Dhaka              | -54.62      |
| Namapara, Khilkhet, Dhaka                      | -62.64      |
| Section 6, Mirpur, Dhaka                       | -87.72      |
| Block J, Baridhara, Dhaka                      | -105.43     |
| Banani Dohs, Dhaka                             | -112.49     |
| D. I. T. Project, Badda, Dhaka                 | -116.14     |
| Block G, Bashundhara R-A, Dhaka                | -169.27     |
| Uttara, Dhaka                                  | -345.09     |
| Sector 18, Uttara, Dhaka                       | -476.8      |
| Bochila, Mohammadpur, Dhaka                    | -15716.22   |
| Block B, Bashundhara R-A, Dhaka                | -2365159.01 |

> Locations that has not even 5 dataset!

| Location                                | Score |
| --------------------------------------- | ----- |
| 1St Colony, Mirpur, Dhaka               | N/A   |
| Arambag Residential Area, Mirpur, Dhaka | N/A   |
| Azimpur, Lalbagh, Dhaka                 | N/A   |
| Bakshi Bazar, Lalbagh, Dhaka            | N/A   |
| Banasree, Dhaka                         | N/A   |
| Bangshal, Dhaka                         | N/A   |
| Barontek, Cantonment, Dhaka             | N/A   |
| Bepari Para, Khilkhet, Dhaka            | N/A   |
| Block A, Niketan, Dhaka                 | N/A   |
| Block B, Aftab Nagar, Dhaka             | N/A   |
| Block B, Banasree, Dhaka                | N/A   |
| Block B, Niketan, Dhaka                 | N/A   |
| Block C, Khilgaon, Dhaka                | N/A   |
| Block C, Niketan, Dhaka                 | N/A   |
| Block D, Lalmatia, Dhaka                | N/A   |
| Block D, Niketan, Dhaka                 | N/A   |
| Block E, Bashundhara R-A, Dhaka         | N/A   |
| Block E, Lalmatia, Dhaka                | N/A   |
| Block F, Aftab Nagar, Dhaka             | N/A   |
| Block F, Banasree, Dhaka                | N/A   |
| Block F, Lalmatia, Dhaka                | N/A   |
| Block G, Aftab Nagar, Dhaka             | N/A   |
| Block H, Aftab Nagar, Dhaka             | N/A   |
| Block J, Banasree, Dhaka                | N/A   |
| Block M, Banasree, Dhaka                | N/A   |
| Boro Maghbazar, Maghbazar, Dhaka        | N/A   |
| Chalabon, Dakshin Khan, Dhaka           | N/A   |
| Chowdhuripara, Khilgaon, Dhaka          | N/A   |
| Comfort Housing, Adabor, Dhaka          | N/A   |
| Dhaka Uddan, Mohammadpur, Dhaka         | N/A   |
| East Badda, Badda, Dhaka                | N/A   |
| East Nakhalpara, Tejgaon, Dhaka         | N/A   |
| Farmgate, Tejgaon, Dhaka                | N/A   |
| Gojmohal, Hazaribag, Dhaka              | N/A   |
| Goltek, Cantonment, Dhaka               | N/A   |
| Goran, Khilgaon, Dhaka                  | N/A   |
| Indira Road, Tejgaon, Dhaka             | N/A   |
| Jagannathpur, Badda, Dhaka              | N/A   |
| Jahuri Moholla, Mohammadpur, Dhaka      | N/A   |
| Jamtola, Khilkhet, Dhaka                | N/A   |
| Kachukhet, Dhaka                        | N/A   |
| Kafrul, Dhaka                           | N/A   |
| Kamarpara, Turag, Dhaka                 | N/A   |
| Katabon, New Market, Dhaka              | N/A   |
| Kha Para, Khilkhet, Dhaka               | N/A   |
| Khilbari Tek, Badda, Dhaka              | N/A   |
| Kuril, Dhaka                            | N/A   |
| Lalmatia, Dhaka                         | N/A   |
| Meradia, Khilgaon, Dhaka                | N/A   |
| Middle Bashabo, Bashabo, Dhaka          | N/A   |
| Middle Monipur, Mirpur, Dhaka           | N/A   |
| Mirpur Road, New Market, Dhaka          | N/A   |
| Moddo Para, Khilkhet, Dhaka             | N/A   |
| Mohakhali, Dhaka                        | N/A   |
| Moneshwar, Hazaribag, Dhaka             | N/A   |
| Monipur, Mirpur, Dhaka                  | N/A   |
| Monipuripara, Tejgaon, Dhaka            | N/A   |
| Moushair, Dakshin Khan, Dhaka           | N/A   |
| Naddapara, Dakshin Khan, Dhaka          | N/A   |
| Natun Bazar, Badda, Dhaka               | N/A   |
| Naya Nagar, Khilkhet, Dhaka             | N/A   |
| New Eskaton, Eskaton, Dhaka             | N/A   |
| Niketan, Dhaka                          | N/A   |
| Nikunja 1, Nikunja, Dhaka               | N/A   |
| North Adabor, Adabor, Dhaka             | N/A   |
| North Azampur, Dakshin Khan, Dhaka      | N/A   |
| North Shahjahanpur, Dhaka               | N/A   |
| North Shahjahanpur, Shahjahanpur, Dhaka | N/A   |
| Nowapara, Dakshin Khan, Dhaka           | N/A   |
| Old Eskaton, Eskaton, Dhaka             | N/A   |
| Paribagh, Dhaka                         | N/A   |
| Pink City, Dhaka                        | N/A   |
| Purana Paltan, Motijheel, Dhaka         | N/A   |
| Rampura, Dhaka                          | N/A   |
| Rayer Bazaar, Hazaribag, Dhaka          | N/A   |
| Ring Road, Shyamoli, Dhaka              | N/A   |
| Rupnagar R/A, Mirpur, Dhaka             | N/A   |
| Sarulia, Demra, Dhaka                   | N/A   |
| School Road, Mohakhali, Dhaka           | N/A   |
| Section 15, Mirpur, Dhaka               | N/A   |
| Section 5, Mirpur, Dhaka                | N/A   |
| Section 7, Mirpur, Dhaka                | N/A   |
| Shahbagh, Dhaka                         | N/A   |
| Shahjadpur, Badda, Dhaka                | N/A   |
| Shantibag, Malibagh, Dhaka              | N/A   |
| Sher-E-Bangla Road, Dhanmondi, Dhaka    | N/A   |
| Shukrabad, Dhanmondi, Dhaka             | N/A   |
| South Bashabo, Bashabo, Dhaka           | N/A   |
| South Chalabon, Dakshin Khan, Dhaka     | N/A   |
| South Jatra Bari, Jatra Bari, Dhaka     | N/A   |
| Sunibir Housing Society, Adabor, Dhaka  | N/A   |
| Taltola, Dakshin Khan, Dhaka            | N/A   |
| Taltola, Dhaka                          | N/A   |
| Tejgaon Industrial Area, Tejgaon, Dhaka | N/A   |
| Tejkunipara, Tejgaon, Dhaka             | N/A   |
| Tejturi Bazar, Tejgaon, Dhaka           | N/A   |
| Tikatuli, Sutrapur, Dhaka               | N/A   |
| Ullan, Rampura, Dhaka                   | N/A   |
| Uttar Para, Khilkhet, Dhaka             | N/A   |
| Vashantek, Cantonment, Dhaka            | N/A   |
| Wari, Sutrapur, Dhaka                   | N/A   |
| West Dhanmondi, Dhanmondi, Dhaka        | N/A   |
| West Kafrul, Taltola, Dhaka             | N/A   |
| West Kalachandpur, Kalachandpur, Dhaka  | N/A   |
| West Mollartek, Dakshin Khan, Dhaka     | N/A   |
| West Monipur, Mirpur, Dhaka             | N/A   |
| Zafrabad, Dhaka                         | N/A   |
| Zafrabad, Hazaribag, Dhaka              | N/A   |
