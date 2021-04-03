# FedPAQ-MNIST-implemenation
An implementation of FedPAQ using different experimental parameters it does not entail the actual paper implementation but rather my own implementation, making it simpler to understand. We will be looking at different variations of how, r(number of clients to be selected), t (local epochs) and precision (Quantizer levels).

Here, we clearly understand the importance of `local epochs` being between 1 and root(T) (here, 10). We can see that best `local epoch` is `3`, which is the closest root(10), at the same time, when comparing 1 and 5, we see 1 having a better performance and hence containing learning potential. Thereby, experimentally supporting the claim that the ideal, local number of epochs should be between `1` and `root(T)` (or root(10) ~ 3).

As for the value of `r` I used ratios with respect to the total number of clients in this implementation. This was done for easier and readable implementation. We also see an incline with the increase in `r`, as `r` increases from `0.5` to `1`. FedPAQ's claim that ideal performance will be given by `r = 100%` is supported.

Finally quantization is enabled in the form precision here, so as to allow beginners and non-coders to understand. This is still supported by the FedPAQ's claims with regards to Quantizer Levels (`s`). As `precision` increases, so does performance.

## Best-Perfomance:

Best Performance was given by:

local_epochs | r | precision | train_loss | train_acc | test_loss | test_acc
---|---|---|---|---|---|---
3.0 | 1.0 | 7.0 | 1.50859 | 0.95441 | 1.51419| 0.94875

## Analysis-of-Variations:

Some images to demonstrates - 
* local epochs (fixing precision and r to 7 and 100% respectively)
* r (fixing local_epochs and precision to 3 and 7 respectively)
* precision (fixing local_epochs and r to 3 and 100% respectively)

#### local epochs:
![local-epochs](https://github.com/AmanPriyanshu/FedPAQ-MNIST-implemenation/blob/main/raw_images/local_epochs_variations.png)
#### r:
![r](https://github.com/AmanPriyanshu/FedPAQ-MNIST-implemenation/blob/main/raw_images/r_variations.png)
#### precision:
![precision](https://github.com/AmanPriyanshu/FedPAQ-MNIST-implemenation/blob/main/raw_images/precision_variations.png)

## Analysis-Synopsis:

local_epochs | r | precision | train_loss | train_acc | test_loss | test_acc
---|---|---|---|---|---|---
1.0 | 0.5 | 5.0 | 1.728125858234196 | 0.4895436356707317 | 1.7305705044247688 | 0.4900372706422018
1.0 | 0.5 | 6.0 | 1.6821167879715198 | 0.6466987423780488 | 1.682835039742496 | 0.6474340596330275
1.0 | 0.5 | 7.0 | 1.9282506533512256 | 0.18576124237804878 | 1.9297028440947925 | 0.19190797018348624
1.0 | 0.667 | 5.0 | 1.9637017875182918 | 0.18278391768292682 | 1.9619540036271472 | 0.1885751146788991
1.0 | 0.667 | 6.0 | 1.6140101526568575 | 0.7112233231707317 | 1.6153453304133285 | 0.7158113532110092
1.0 | 0.667 | 7.0 | 1.670547209861802 | 0.5715987042682927 | 1.6725489371413484 | 0.5724268922018348
1.0 | 0.833 | 5.0 | 1.568389562935364 | 0.8054258765243902 | 1.5698062525976688 | 0.801963876146789
1.0 | 0.833 | 6.0 | 1.5451831835799101 | 0.9042016006097561 | 1.54767341570023 | 0.9031321674311926
1.0 | 0.833 | 7.0 | 1.5589591315606746 | 0.8348656631097561 | 1.5625734509678062 | 0.8296301605504587
1.0 | 1.0 | 5.0 | 2.3012757584816073 | 0.11149485518292683 | 2.3010153857939835 | 0.11374713302752294
1.0 | 1.0 | 6.0 | 1.5486011079898694 | 0.8852419969512195 | 1.5528842708386412 | 0.8799813646788991
1.0 | 1.0 | 7.0 | 1.8473397918590686 | 0.20855564024390244 | 1.8505481497957073 | 0.21115252293577982
3.0 | 0.5 | 5.0 | 1.5333226115965262 | 0.9275914634146342 | 1.5364157256730107 | 0.9254945527522935
3.0 | 0.5 | 6.0 | 1.5398044862398288 | 0.9187547637195121 | 1.545823039264854 | 0.9133815940366973
3.0 | 0.5 | 7.0 | 1.5357685688792206 | 0.9268530868902439 | 1.5398197699030605 | 0.9238102064220184
3.0 | 0.667 | 5.0 | 1.5232759583287123 | 0.9389529344512195 | 1.5291103157428427 | 0.9348480504587156
3.0 | 0.667 | 6.0 | 1.5267741320336736 | 0.9351895960365854 | 1.5295108359888059 | 0.9338446100917431
3.0 | 0.667 | 7.0 | 1.5265251012837016 | 0.9386909298780488 | 1.5320419450418665 | 0.9342029816513762
3.0 | 0.833 | 5.0 | 1.5121963576572697 | 0.9512195121951219 | 1.5180684617899973 | 0.9461725917431193
3.0 | 0.833 | 6.0 | 1.5153437304060633 | 0.9490043826219512 | 1.5206277064227183 | 0.9426605504587156
3.0 | 0.833 | 7.0 | 1.521840628690836 | 0.9415015243902439 | 1.5250294640523578 | 0.9368907683486238
3.0 | 1.0 | 5.0 | 1.5163829359339505 | 0.9474561737804879 | 1.5226867729370748 | 0.9413345756880734
3.0 | 1.0 | 6.0 | 1.51277627959484 | 0.9515291539634146 | 1.5181890201131139 | 0.9465668004587156
3.0 | 1.0 | 7.0 | 1.5085993612684854 | 0.9544112042682927 | 1.514190214489578 | 0.9487528669724771
5.0 | 0.5 | 5.0 | 1.5281373287846403 | 0.9335461128048781 | 1.5311877705635282 | 0.9305475917431193
5.0 | 0.5 | 6.0 | 1.5159811410235196 | 0.9471227134146342 | 1.519424247632333 | 0.9446674311926605
5.0 | 0.5 | 7.0 | 1.5160528933856545 | 0.9477896341463414 | 1.5204480607575233 | 0.9433414564220184
5.0 | 0.667 | 5.0 | 1.511145905023668 | 0.9517197027439024 | 1.5159289875161757 | 0.9467459862385321
5.0 | 0.667 | 6.0 | 1.5198594655205564 | 0.9443835746951219 | 1.5241264225146092 | 0.9396860665137615
5.0 | 0.667 | 7.0 | 1.5329274164467324 | 0.9320455411585366 | 1.538827650590774 | 0.9260321100917431
5.0 | 0.833 | 5.0 | 1.51707308648563 | 0.9463366996951219 | 1.5228457964888407 | 0.941047878440367
5.0 | 0.833 | 6.0 | 1.5750223732576139 | 0.8799304496951219 | 1.5786832882723678 | 0.876182626146789
5.0 | 0.833 | 7.0 | 1.5106694149534876 | 0.9526486280487805 | 1.516274865614165 | 0.946495126146789
5.0 | 1.0 | 5.0 | 1.521039308207791 | 0.9413109756097561 | 1.5255555472242723 | 0.9377866972477065
5.0 | 1.0 | 6.0 | 1.5142119323335044 | 0.9487661966463414 | 1.519685910382402 | 0.9436998279816514
5.0 | 1.0 | 7.0 | 2.3012793035041996 | 0.11149485518292683 | 2.301147645766582 | 0.11374713302752294

### Reference:
```console
@ARTICLE{2019arXiv190913014R,
       author = {{Reisizadeh}, Amirhossein and {Mokhtari}, Aryan and {Hassani}, Hamed and {Jadbabaie}, Ali and {Pedarsani}, Ramtin},
        title = "{FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Distributed, Parallel, and Cluster Computing, Mathematics - Optimization and Control, Statistics - Machine Learning},
         year = 2019,
        month = sep,
          eid = {arXiv:1909.13014},
        pages = {arXiv:1909.13014},
archivePrefix = {arXiv},
       eprint = {1909.13014},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190913014R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
