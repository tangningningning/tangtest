# Cache Q representation and target diagnostic

## A. Dataset reuse and integrity

- Dataset schema valid: `True`
- Candidate alignment: `True`
- Split: train `(1, 2, 3, 4, 5, 6, 7, 8)`, validation `(9, 10)`, held-out `(15, 20, 25, 30)`.
- No environment, Matching, KKT, PPO, or new realization was executed.

## B. Current architecture memorization probe

```json
{
  "name": "MEMORIZATION",
  "target_kind": "residual",
  "architecture": "cache_action_conditioned_residual_q_mlp",
  "architecture_version": 1,
  "fresh_initialization": true,
  "hidden_dim": 64,
  "max_epochs": 2000,
  "patience": null,
  "early_stopping": false,
  "best_epoch": 2000,
  "epochs_executed": 2000,
  "best_validation_loss": null,
  "train_loss_initial": 0.9670089483261108,
  "train_loss_final": 1.7956006862126134e-05,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.9670089483261108,
      "validation_loss": null
    },
    {
      "epoch": 2,
      "train_loss": 0.7670729094081454,
      "validation_loss": null
    },
    {
      "epoch": 3,
      "train_loss": 0.6397746470239427,
      "validation_loss": null
    },
    {
      "epoch": 4,
      "train_loss": 0.5567716525660621,
      "validation_loss": null
    },
    {
      "epoch": 5,
      "train_loss": 0.4485811922285292,
      "validation_loss": null
    },
    {
      "epoch": 6,
      "train_loss": 0.4411153594652812,
      "validation_loss": null
    },
    {
      "epoch": 7,
      "train_loss": 0.3677063352531857,
      "validation_loss": null
    },
    {
      "epoch": 8,
      "train_loss": 0.32618027925491333,
      "validation_loss": null
    },
    {
      "epoch": 9,
      "train_loss": 0.32080403963724774,
      "validation_loss": null
    },
    {
      "epoch": 10,
      "train_loss": 0.25852569772137535,
      "validation_loss": null
    },
    {
      "epoch": 11,
      "train_loss": 0.2305315997865465,
      "validation_loss": null
    },
    {
      "epoch": 12,
      "train_loss": 0.23752027087741429,
      "validation_loss": null
    },
    {
      "epoch": 13,
      "train_loss": 0.22230189541975656,
      "validation_loss": null
    },
    {
      "epoch": 14,
      "train_loss": 0.20455306437280443,
      "validation_loss": null
    },
    {
      "epoch": 15,
      "train_loss": 0.19769109785556793,
      "validation_loss": null
    },
    {
      "epoch": 16,
      "train_loss": 0.19250747230317858,
      "validation_loss": null
    },
    {
      "epoch": 17,
      "train_loss": 0.19798420038488176,
      "validation_loss": null
    },
    {
      "epoch": 18,
      "train_loss": 0.19907689756817287,
      "validation_loss": null
    },
    {
      "epoch": 19,
      "train_loss": 0.1796902683046129,
      "validation_loss": null
    },
    {
      "epoch": 20,
      "train_loss": 0.17688763638337454,
      "validation_loss": null
    },
    {
      "epoch": 21,
      "train_loss": 0.18305565085675982,
      "validation_loss": null
    },
    {
      "epoch": 22,
      "train_loss": 0.18512911515103447,
      "validation_loss": null
    },
    {
      "epoch": 23,
      "train_loss": 0.1724550930990113,
      "validation_loss": null
    },
    {
      "epoch": 24,
      "train_loss": 0.160256157318751,
      "validation_loss": null
    },
    {
      "epoch": 25,
      "train_loss": 0.1605328776770168,
      "validation_loss": null
    },
    {
      "epoch": 26,
      "train_loss": 0.16829904086059994,
      "validation_loss": null
    },
    {
      "epoch": 27,
      "train_loss": 0.15627859036127725,
      "validation_loss": null
    },
    {
      "epoch": 28,
      "train_loss": 0.15462456478012931,
      "validation_loss": null
    },
    {
      "epoch": 29,
      "train_loss": 0.15146917601426443,
      "validation_loss": null
    },
    {
      "epoch": 30,
      "train_loss": 0.15615988191631106,
      "validation_loss": null
    },
    {
      "epoch": 31,
      "train_loss": 0.19299837946891785,
      "validation_loss": null
    },
    {
      "epoch": 32,
      "train_loss": 0.15251764158407846,
      "validation_loss": null
    },
    {
      "epoch": 33,
      "train_loss": 0.14524815645482805,
      "validation_loss": null
    },
    {
      "epoch": 34,
      "train_loss": 0.15962019066015878,
      "validation_loss": null
    },
    {
      "epoch": 35,
      "train_loss": 0.14814596705966526,
      "validation_loss": null
    },
    {
      "epoch": 36,
      "train_loss": 0.12770194808642069,
      "validation_loss": null
    },
    {
      "epoch": 37,
      "train_loss": 0.1349778100848198,
      "validation_loss": null
    },
    {
      "epoch": 38,
      "train_loss": 0.13885611626836988,
      "validation_loss": null
    },
    {
      "epoch": 39,
      "train_loss": 0.13295306513706842,
      "validation_loss": null
    },
    {
      "epoch": 40,
      "train_loss": 0.1231121313240793,
      "validation_loss": null
    },
    {
      "epoch": 41,
      "train_loss": 0.11444011661741468,
      "validation_loss": null
    },
    {
      "epoch": 42,
      "train_loss": 0.11058716475963593,
      "validation_loss": null
    },
    {
      "epoch": 43,
      "train_loss": 0.10425048569838206,
      "validation_loss": null
    },
    {
      "epoch": 44,
      "train_loss": 0.09571451817949612,
      "validation_loss": null
    },
    {
      "epoch": 45,
      "train_loss": 0.09663522243499756,
      "validation_loss": null
    },
    {
      "epoch": 46,
      "train_loss": 0.10284182677666347,
      "validation_loss": null
    },
    {
      "epoch": 47,
      "train_loss": 0.10865749998225106,
      "validation_loss": null
    },
    {
      "epoch": 48,
      "train_loss": 0.08189273832572831,
      "validation_loss": null
    },
    {
      "epoch": 49,
      "train_loss": 0.08144650980830193,
      "validation_loss": null
    },
    {
      "epoch": 50,
      "train_loss": 0.07454134316907989,
      "validation_loss": null
    },
    {
      "epoch": 51,
      "train_loss": 0.06805343594815996,
      "validation_loss": null
    },
    {
      "epoch": 52,
      "train_loss": 0.06729855595363511,
      "validation_loss": null
    },
    {
      "epoch": 53,
      "train_loss": 0.0661299570153157,
      "validation_loss": null
    },
    {
      "epoch": 54,
      "train_loss": 0.06774844312005573,
      "validation_loss": null
    },
    {
      "epoch": 55,
      "train_loss": 0.07193780193726222,
      "validation_loss": null
    },
    {
      "epoch": 56,
      "train_loss": 0.06278875097632408,
      "validation_loss": null
    },
    {
      "epoch": 57,
      "train_loss": 0.058146920055150986,
      "validation_loss": null
    },
    {
      "epoch": 58,
      "train_loss": 0.05013245820171303,
      "validation_loss": null
    },
    {
      "epoch": 59,
      "train_loss": 0.049704051266113915,
      "validation_loss": null
    },
    {
      "epoch": 60,
      "train_loss": 0.04728428481353654,
      "validation_loss": null
    },
    {
      "epoch": 61,
      "train_loss": 0.04812609197364913,
      "validation_loss": null
    },
    {
      "epoch": 62,
      "train_loss": 0.04410661322375139,
      "validation_loss": null
    },
    {
      "epoch": 63,
      "train_loss": 0.04721204853720135,
      "validation_loss": null
    },
    {
      "epoch": 64,
      "train_loss": 0.046985445337163076,
      "validation_loss": null
    },
    {
      "epoch": 65,
      "train_loss": 0.04524014352096452,
      "validation_loss": null
    },
    {
      "epoch": 66,
      "train_loss": 0.041195545345544815,
      "validation_loss": null
    },
    {
      "epoch": 67,
      "train_loss": 0.052820984895030655,
      "validation_loss": null
    },
    {
      "epoch": 68,
      "train_loss": 0.050937080962790385,
      "validation_loss": null
    },
    {
      "epoch": 69,
      "train_loss": 0.04093843595021301,
      "validation_loss": null
    },
    {
      "epoch": 70,
      "train_loss": 0.046564868133929044,
      "validation_loss": null
    },
    {
      "epoch": 71,
      "train_loss": 0.04135860544111994,
      "validation_loss": null
    },
    {
      "epoch": 72,
      "train_loss": 0.054086744578348264,
      "validation_loss": null
    },
    {
      "epoch": 73,
      "train_loss": 0.05222185100946161,
      "validation_loss": null
    },
    {
      "epoch": 74,
      "train_loss": 0.03812821002470122,
      "validation_loss": null
    },
    {
      "epoch": 75,
      "train_loss": 0.03666503189338578,
      "validation_loss": null
    },
    {
      "epoch": 76,
      "train_loss": 0.03155734659069114,
      "validation_loss": null
    },
    {
      "epoch": 77,
      "train_loss": 0.031430195603105754,
      "validation_loss": null
    },
    {
      "epoch": 78,
      "train_loss": 0.03148655821051863,
      "validation_loss": null
    },
    {
      "epoch": 79,
      "train_loss": 0.031615350395441055,
      "validation_loss": null
    },
    {
      "epoch": 80,
      "train_loss": 0.03153836768534449,
      "validation_loss": null
    },
    {
      "epoch": 81,
      "train_loss": 0.02642695088353422,
      "validation_loss": null
    },
    {
      "epoch": 82,
      "train_loss": 0.025865680641598172,
      "validation_loss": null
    },
    {
      "epoch": 83,
      "train_loss": 0.024947799742221832,
      "validation_loss": null
    },
    {
      "epoch": 84,
      "train_loss": 0.024123871905936137,
      "validation_loss": null
    },
    {
      "epoch": 85,
      "train_loss": 0.027370489098959498,
      "validation_loss": null
    },
    {
      "epoch": 86,
      "train_loss": 0.026989666124184925,
      "validation_loss": null
    },
    {
      "epoch": 87,
      "train_loss": 0.023500378140144877,
      "validation_loss": null
    },
    {
      "epoch": 88,
      "train_loss": 0.021930328156385157,
      "validation_loss": null
    },
    {
      "epoch": 89,
      "train_loss": 0.022925008709232014,
      "validation_loss": null
    },
    {
      "epoch": 90,
      "train_loss": 0.02071332114024295,
      "validation_loss": null
    },
    {
      "epoch": 91,
      "train_loss": 0.022573464653558202,
      "validation_loss": null
    },
    {
      "epoch": 92,
      "train_loss": 0.021998967561456893,
      "validation_loss": null
    },
    {
      "epoch": 93,
      "train_loss": 0.021163043255607288,
      "validation_loss": null
    },
    {
      "epoch": 94,
      "train_loss": 0.02141553981022702,
      "validation_loss": null
    },
    {
      "epoch": 95,
      "train_loss": 0.020925541408360004,
      "validation_loss": null
    },
    {
      "epoch": 96,
      "train_loss": 0.021114869353671867,
      "validation_loss": null
    },
    {
      "epoch": 97,
      "train_loss": 0.019965145426491897,
      "validation_loss": null
    },
    {
      "epoch": 98,
      "train_loss": 0.019007984548807144,
      "validation_loss": null
    },
    {
      "epoch": 99,
      "train_loss": 0.01937240579475959,
      "validation_loss": null
    },
    {
      "epoch": 100,
      "train_loss": 0.018589054958687887,
      "validation_loss": null
    },
    {
      "epoch": 101,
      "train_loss": 0.018848366828428373,
      "validation_loss": null
    },
    {
      "epoch": 102,
      "train_loss": 0.01975679263058636,
      "validation_loss": null
    },
    {
      "epoch": 103,
      "train_loss": 0.019532447163429525,
      "validation_loss": null
    },
    {
      "epoch": 104,
      "train_loss": 0.020178259143398866,
      "validation_loss": null
    },
    {
      "epoch": 105,
      "train_loss": 0.01863332682599624,
      "validation_loss": null
    },
    {
      "epoch": 106,
      "train_loss": 0.02001560479402542,
      "validation_loss": null
    },
    {
      "epoch": 107,
      "train_loss": 0.017364645790722635,
      "validation_loss": null
    },
    {
      "epoch": 108,
      "train_loss": 0.018319587430192366,
      "validation_loss": null
    },
    {
      "epoch": 109,
      "train_loss": 0.021371898448301688,
      "validation_loss": null
    },
    {
      "epoch": 110,
      "train_loss": 0.0202659766914116,
      "validation_loss": null
    },
    {
      "epoch": 111,
      "train_loss": 0.023180602221853204,
      "validation_loss": null
    },
    {
      "epoch": 112,
      "train_loss": 0.01886789407581091,
      "validation_loss": null
    },
    {
      "epoch": 113,
      "train_loss": 0.01868450227710936,
      "validation_loss": null
    },
    {
      "epoch": 114,
      "train_loss": 0.0164801732947429,
      "validation_loss": null
    },
    {
      "epoch": 115,
      "train_loss": 0.01548540530105432,
      "validation_loss": null
    },
    {
      "epoch": 116,
      "train_loss": 0.020822276671727497,
      "validation_loss": null
    },
    {
      "epoch": 117,
      "train_loss": 0.015963049605488777,
      "validation_loss": null
    },
    {
      "epoch": 118,
      "train_loss": 0.014386724370221296,
      "validation_loss": null
    },
    {
      "epoch": 119,
      "train_loss": 0.013235284946858883,
      "validation_loss": null
    },
    {
      "epoch": 120,
      "train_loss": 0.013511982435981432,
      "validation_loss": null
    },
    {
      "epoch": 121,
      "train_loss": 0.013734043886264166,
      "validation_loss": null
    },
    {
      "epoch": 122,
      "train_loss": 0.013703956993089782,
      "validation_loss": null
    },
    {
      "epoch": 123,
      "train_loss": 0.013630891839663187,
      "validation_loss": null
    },
    {
      "epoch": 124,
      "train_loss": 0.015523516366051303,
      "validation_loss": null
    },
    {
      "epoch": 125,
      "train_loss": 0.013794054070280658,
      "validation_loss": null
    },
    {
      "epoch": 126,
      "train_loss": 0.013682605595224433,
      "validation_loss": null
    },
    {
      "epoch": 127,
      "train_loss": 0.015767277632322576,
      "validation_loss": null
    },
    {
      "epoch": 128,
      "train_loss": 0.01792887256791194,
      "validation_loss": null
    },
    {
      "epoch": 129,
      "train_loss": 0.014814612352185778,
      "validation_loss": null
    },
    {
      "epoch": 130,
      "train_loss": 0.012178102663407723,
      "validation_loss": null
    },
    {
      "epoch": 131,
      "train_loss": 0.012264126704798805,
      "validation_loss": null
    },
    {
      "epoch": 132,
      "train_loss": 0.01473305436472098,
      "validation_loss": null
    },
    {
      "epoch": 133,
      "train_loss": 0.013892419946690401,
      "validation_loss": null
    },
    {
      "epoch": 134,
      "train_loss": 0.01196768455621269,
      "validation_loss": null
    },
    {
      "epoch": 135,
      "train_loss": 0.01107034252749549,
      "validation_loss": null
    },
    {
      "epoch": 136,
      "train_loss": 0.01176156915931238,
      "validation_loss": null
    },
    {
      "epoch": 137,
      "train_loss": 0.011401556535727449,
      "validation_loss": null
    },
    {
      "epoch": 138,
      "train_loss": 0.01157102764894565,
      "validation_loss": null
    },
    {
      "epoch": 139,
      "train_loss": 0.012569615099993017,
      "validation_loss": null
    },
    {
      "epoch": 140,
      "train_loss": 0.011494936162812842,
      "validation_loss": null
    },
    {
      "epoch": 141,
      "train_loss": 0.011688815067625709,
      "validation_loss": null
    },
    {
      "epoch": 142,
      "train_loss": 0.011978824622929096,
      "validation_loss": null
    },
    {
      "epoch": 143,
      "train_loss": 0.010541717263145579,
      "validation_loss": null
    },
    {
      "epoch": 144,
      "train_loss": 0.010689213613255156,
      "validation_loss": null
    },
    {
      "epoch": 145,
      "train_loss": 0.013049246898541847,
      "validation_loss": null
    },
    {
      "epoch": 146,
      "train_loss": 0.012474879260278411,
      "validation_loss": null
    },
    {
      "epoch": 147,
      "train_loss": 0.012849756671736637,
      "validation_loss": null
    },
    {
      "epoch": 148,
      "train_loss": 0.014532870509558253,
      "validation_loss": null
    },
    {
      "epoch": 149,
      "train_loss": 0.012505971723132663,
      "validation_loss": null
    },
    {
      "epoch": 150,
      "train_loss": 0.01198006711072392,
      "validation_loss": null
    },
    {
      "epoch": 151,
      "train_loss": 0.011442326733635532,
      "validation_loss": null
    },
    {
      "epoch": 152,
      "train_loss": 0.011960472290714582,
      "validation_loss": null
    },
    {
      "epoch": 153,
      "train_loss": 0.01134645887133148,
      "validation_loss": null
    },
    {
      "epoch": 154,
      "train_loss": 0.011507675051689148,
      "validation_loss": null
    },
    {
      "epoch": 155,
      "train_loss": 0.008445735451661877,
      "validation_loss": null
    },
    {
      "epoch": 156,
      "train_loss": 0.008742169477045536,
      "validation_loss": null
    },
    {
      "epoch": 157,
      "train_loss": 0.009532255586236715,
      "validation_loss": null
    },
    {
      "epoch": 158,
      "train_loss": 0.009314872014025847,
      "validation_loss": null
    },
    {
      "epoch": 159,
      "train_loss": 0.008982425245145956,
      "validation_loss": null
    },
    {
      "epoch": 160,
      "train_loss": 0.008291125970168246,
      "validation_loss": null
    },
    {
      "epoch": 161,
      "train_loss": 0.008892763561258713,
      "validation_loss": null
    },
    {
      "epoch": 162,
      "train_loss": 0.00840774199201001,
      "validation_loss": null
    },
    {
      "epoch": 163,
      "train_loss": 0.0083930897526443,
      "validation_loss": null
    },
    {
      "epoch": 164,
      "train_loss": 0.008382867142144177,
      "validation_loss": null
    },
    {
      "epoch": 165,
      "train_loss": 0.009846008589698209,
      "validation_loss": null
    },
    {
      "epoch": 166,
      "train_loss": 0.009435781329456303,
      "validation_loss": null
    },
    {
      "epoch": 167,
      "train_loss": 0.010628958077480396,
      "validation_loss": null
    },
    {
      "epoch": 168,
      "train_loss": 0.010668049462967448,
      "validation_loss": null
    },
    {
      "epoch": 169,
      "train_loss": 0.010145367537107732,
      "validation_loss": null
    },
    {
      "epoch": 170,
      "train_loss": 0.008340941224661138,
      "validation_loss": null
    },
    {
      "epoch": 171,
      "train_loss": 0.00866586042361127,
      "validation_loss": null
    },
    {
      "epoch": 172,
      "train_loss": 0.009169910258303085,
      "validation_loss": null
    },
    {
      "epoch": 173,
      "train_loss": 0.010659020652787553,
      "validation_loss": null
    },
    {
      "epoch": 174,
      "train_loss": 0.009253218381976088,
      "validation_loss": null
    },
    {
      "epoch": 175,
      "train_loss": 0.010350958340697818,
      "validation_loss": null
    },
    {
      "epoch": 176,
      "train_loss": 0.010632520489808586,
      "validation_loss": null
    },
    {
      "epoch": 177,
      "train_loss": 0.009313319757994678,
      "validation_loss": null
    },
    {
      "epoch": 178,
      "train_loss": 0.008433038006640144,
      "validation_loss": null
    },
    {
      "epoch": 179,
      "train_loss": 0.006988023666457998,
      "validation_loss": null
    },
    {
      "epoch": 180,
      "train_loss": 0.007510322249597973,
      "validation_loss": null
    },
    {
      "epoch": 181,
      "train_loss": 0.007566910030113326,
      "validation_loss": null
    },
    {
      "epoch": 182,
      "train_loss": 0.006832044830338823,
      "validation_loss": null
    },
    {
      "epoch": 183,
      "train_loss": 0.007602806616988447,
      "validation_loss": null
    },
    {
      "epoch": 184,
      "train_loss": 0.0063684603406323325,
      "validation_loss": null
    },
    {
      "epoch": 185,
      "train_loss": 0.006513503664690588,
      "validation_loss": null
    },
    {
      "epoch": 186,
      "train_loss": 0.006809731583214468,
      "validation_loss": null
    },
    {
      "epoch": 187,
      "train_loss": 0.008158454826722542,
      "validation_loss": null
    },
    {
      "epoch": 188,
      "train_loss": 0.007570154913183715,
      "validation_loss": null
    },
    {
      "epoch": 189,
      "train_loss": 0.006674864639838536,
      "validation_loss": null
    },
    {
      "epoch": 190,
      "train_loss": 0.007259312137547467,
      "validation_loss": null
    },
    {
      "epoch": 191,
      "train_loss": 0.0061309670305086505,
      "validation_loss": null
    },
    {
      "epoch": 192,
      "train_loss": 0.006097587498111857,
      "validation_loss": null
    },
    {
      "epoch": 193,
      "train_loss": 0.005958445680638154,
      "validation_loss": null
    },
    {
      "epoch": 194,
      "train_loss": 0.006172754957030217,
      "validation_loss": null
    },
    {
      "epoch": 195,
      "train_loss": 0.006316905323829915,
      "validation_loss": null
    },
    {
      "epoch": 196,
      "train_loss": 0.006731854829316338,
      "validation_loss": null
    },
    {
      "epoch": 197,
      "train_loss": 0.007426472846418619,
      "validation_loss": null
    },
    {
      "epoch": 198,
      "train_loss": 0.0077802410038808984,
      "validation_loss": null
    },
    {
      "epoch": 199,
      "train_loss": 0.00644134768905739,
      "validation_loss": null
    },
    {
      "epoch": 200,
      "train_loss": 0.006436162524753147,
      "validation_loss": null
    },
    {
      "epoch": 201,
      "train_loss": 0.00573023920878768,
      "validation_loss": null
    },
    {
      "epoch": 202,
      "train_loss": 0.00810047321849399,
      "validation_loss": null
    },
    {
      "epoch": 203,
      "train_loss": 0.008267865723205937,
      "validation_loss": null
    },
    {
      "epoch": 204,
      "train_loss": 0.006317234908541043,
      "validation_loss": null
    },
    {
      "epoch": 205,
      "train_loss": 0.006324374582618475,
      "validation_loss": null
    },
    {
      "epoch": 206,
      "train_loss": 0.007340948097407818,
      "validation_loss": null
    },
    {
      "epoch": 207,
      "train_loss": 0.00602115325940152,
      "validation_loss": null
    },
    {
      "epoch": 208,
      "train_loss": 0.005504426675745183,
      "validation_loss": null
    },
    {
      "epoch": 209,
      "train_loss": 0.006208185003035598,
      "validation_loss": null
    },
    {
      "epoch": 210,
      "train_loss": 0.005944869584507412,
      "validation_loss": null
    },
    {
      "epoch": 211,
      "train_loss": 0.00615455383538372,
      "validation_loss": null
    },
    {
      "epoch": 212,
      "train_loss": 0.00541230793007546,
      "validation_loss": null
    },
    {
      "epoch": 213,
      "train_loss": 0.005748634252490269,
      "validation_loss": null
    },
    {
      "epoch": 214,
      "train_loss": 0.005528693863501151,
      "validation_loss": null
    },
    {
      "epoch": 215,
      "train_loss": 0.005211868866657217,
      "validation_loss": null
    },
    {
      "epoch": 216,
      "train_loss": 0.005284884504766928,
      "validation_loss": null
    },
    {
      "epoch": 217,
      "train_loss": 0.005059840726769633,
      "validation_loss": null
    },
    {
      "epoch": 218,
      "train_loss": 0.005144509163478183,
      "validation_loss": null
    },
    {
      "epoch": 219,
      "train_loss": 0.005142968096252944,
      "validation_loss": null
    },
    {
      "epoch": 220,
      "train_loss": 0.004642987209889624,
      "validation_loss": null
    },
    {
      "epoch": 221,
      "train_loss": 0.0049651936990105444,
      "validation_loss": null
    },
    {
      "epoch": 222,
      "train_loss": 0.004874922438628144,
      "validation_loss": null
    },
    {
      "epoch": 223,
      "train_loss": 0.005345427068985171,
      "validation_loss": null
    },
    {
      "epoch": 224,
      "train_loss": 0.00636172325660785,
      "validation_loss": null
    },
    {
      "epoch": 225,
      "train_loss": 0.006062461839367946,
      "validation_loss": null
    },
    {
      "epoch": 226,
      "train_loss": 0.00490169747111698,
      "validation_loss": null
    },
    {
      "epoch": 227,
      "train_loss": 0.005362012656405568,
      "validation_loss": null
    },
    {
      "epoch": 228,
      "train_loss": 0.005000171199854877,
      "validation_loss": null
    },
    {
      "epoch": 229,
      "train_loss": 0.004850542431490289,
      "validation_loss": null
    },
    {
      "epoch": 230,
      "train_loss": 0.0056548642460256815,
      "validation_loss": null
    },
    {
      "epoch": 231,
      "train_loss": 0.005438369336641497,
      "validation_loss": null
    },
    {
      "epoch": 232,
      "train_loss": 0.004594932853554686,
      "validation_loss": null
    },
    {
      "epoch": 233,
      "train_loss": 0.0048352516411493225,
      "validation_loss": null
    },
    {
      "epoch": 234,
      "train_loss": 0.004577637287891573,
      "validation_loss": null
    },
    {
      "epoch": 235,
      "train_loss": 0.004987736465409398,
      "validation_loss": null
    },
    {
      "epoch": 236,
      "train_loss": 0.005295567628410127,
      "validation_loss": null
    },
    {
      "epoch": 237,
      "train_loss": 0.008272686455812719,
      "validation_loss": null
    },
    {
      "epoch": 238,
      "train_loss": 0.008072988906254372,
      "validation_loss": null
    },
    {
      "epoch": 239,
      "train_loss": 0.006656998639098472,
      "validation_loss": null
    },
    {
      "epoch": 240,
      "train_loss": 0.005602903871072663,
      "validation_loss": null
    },
    {
      "epoch": 241,
      "train_loss": 0.005497210969527562,
      "validation_loss": null
    },
    {
      "epoch": 242,
      "train_loss": 0.005311663624727064,
      "validation_loss": null
    },
    {
      "epoch": 243,
      "train_loss": 0.005695872733162509,
      "validation_loss": null
    },
    {
      "epoch": 244,
      "train_loss": 0.008051754311761923,
      "validation_loss": null
    },
    {
      "epoch": 245,
      "train_loss": 0.00908012507069442,
      "validation_loss": null
    },
    {
      "epoch": 246,
      "train_loss": 0.007535926764830947,
      "validation_loss": null
    },
    {
      "epoch": 247,
      "train_loss": 0.005254327630003293,
      "validation_loss": null
    },
    {
      "epoch": 248,
      "train_loss": 0.004344782766161693,
      "validation_loss": null
    },
    {
      "epoch": 249,
      "train_loss": 0.0056701479479670525,
      "validation_loss": null
    },
    {
      "epoch": 250,
      "train_loss": 0.00621602893806994,
      "validation_loss": null
    },
    {
      "epoch": 251,
      "train_loss": 0.006146849495255285,
      "validation_loss": null
    },
    {
      "epoch": 252,
      "train_loss": 0.004900948066885273,
      "validation_loss": null
    },
    {
      "epoch": 253,
      "train_loss": 0.005525881528026528,
      "validation_loss": null
    },
    {
      "epoch": 254,
      "train_loss": 0.005985853158765369,
      "validation_loss": null
    },
    {
      "epoch": 255,
      "train_loss": 0.005255166596422593,
      "validation_loss": null
    },
    {
      "epoch": 256,
      "train_loss": 0.007032440923568275,
      "validation_loss": null
    },
    {
      "epoch": 257,
      "train_loss": 0.005621328122085995,
      "validation_loss": null
    },
    {
      "epoch": 258,
      "train_loss": 0.004402915533218119,
      "validation_loss": null
    },
    {
      "epoch": 259,
      "train_loss": 0.0044178051304899985,
      "validation_loss": null
    },
    {
      "epoch": 260,
      "train_loss": 0.003967524687242176,
      "validation_loss": null
    },
    {
      "epoch": 261,
      "train_loss": 0.004400521853110856,
      "validation_loss": null
    },
    {
      "epoch": 262,
      "train_loss": 0.0046803711965266205,
      "validation_loss": null
    },
    {
      "epoch": 263,
      "train_loss": 0.004367965001923342,
      "validation_loss": null
    },
    {
      "epoch": 264,
      "train_loss": 0.0057803211319777704,
      "validation_loss": null
    },
    {
      "epoch": 265,
      "train_loss": 0.004310645809811022,
      "validation_loss": null
    },
    {
      "epoch": 266,
      "train_loss": 0.0044140558327651685,
      "validation_loss": null
    },
    {
      "epoch": 267,
      "train_loss": 0.003963303157231874,
      "validation_loss": null
    },
    {
      "epoch": 268,
      "train_loss": 0.0035935386032279995,
      "validation_loss": null
    },
    {
      "epoch": 269,
      "train_loss": 0.003927141039942701,
      "validation_loss": null
    },
    {
      "epoch": 270,
      "train_loss": 0.0035645000947018466,
      "validation_loss": null
    },
    {
      "epoch": 271,
      "train_loss": 0.003521514370934003,
      "validation_loss": null
    },
    {
      "epoch": 272,
      "train_loss": 0.003447766803825895,
      "validation_loss": null
    },
    {
      "epoch": 273,
      "train_loss": 0.0033987118966049617,
      "validation_loss": null
    },
    {
      "epoch": 274,
      "train_loss": 0.004779089733751284,
      "validation_loss": null
    },
    {
      "epoch": 275,
      "train_loss": 0.004693862594043215,
      "validation_loss": null
    },
    {
      "epoch": 276,
      "train_loss": 0.004684294134171473,
      "validation_loss": null
    },
    {
      "epoch": 277,
      "train_loss": 0.004252709034416411,
      "validation_loss": null
    },
    {
      "epoch": 278,
      "train_loss": 0.0047519145171261495,
      "validation_loss": null
    },
    {
      "epoch": 279,
      "train_loss": 0.003622736626615127,
      "validation_loss": null
    },
    {
      "epoch": 280,
      "train_loss": 0.0034186798923959336,
      "validation_loss": null
    },
    {
      "epoch": 281,
      "train_loss": 0.003359518000959522,
      "validation_loss": null
    },
    {
      "epoch": 282,
      "train_loss": 0.003183217305276129,
      "validation_loss": null
    },
    {
      "epoch": 283,
      "train_loss": 0.0032953118983035288,
      "validation_loss": null
    },
    {
      "epoch": 284,
      "train_loss": 0.004141945262543029,
      "validation_loss": null
    },
    {
      "epoch": 285,
      "train_loss": 0.00416381561404301,
      "validation_loss": null
    },
    {
      "epoch": 286,
      "train_loss": 0.0036360786762088537,
      "validation_loss": null
    },
    {
      "epoch": 287,
      "train_loss": 0.003068833275594645,
      "validation_loss": null
    },
    {
      "epoch": 288,
      "train_loss": 0.0031611286103725433,
      "validation_loss": null
    },
    {
      "epoch": 289,
      "train_loss": 0.003288308675918314,
      "validation_loss": null
    },
    {
      "epoch": 290,
      "train_loss": 0.0038697149914999804,
      "validation_loss": null
    },
    {
      "epoch": 291,
      "train_loss": 0.0038638162788831526,
      "validation_loss": null
    },
    {
      "epoch": 292,
      "train_loss": 0.003247881437548333,
      "validation_loss": null
    },
    {
      "epoch": 293,
      "train_loss": 0.003219559504133132,
      "validation_loss": null
    },
    {
      "epoch": 294,
      "train_loss": 0.0029976346478280094,
      "validation_loss": null
    },
    {
      "epoch": 295,
      "train_loss": 0.00392366795697146,
      "validation_loss": null
    },
    {
      "epoch": 296,
      "train_loss": 0.004007234018192523,
      "validation_loss": null
    },
    {
      "epoch": 297,
      "train_loss": 0.0030556235772868,
      "validation_loss": null
    },
    {
      "epoch": 298,
      "train_loss": 0.0031913434439856145,
      "validation_loss": null
    },
    {
      "epoch": 299,
      "train_loss": 0.003255466516647074,
      "validation_loss": null
    },
    {
      "epoch": 300,
      "train_loss": 0.0034241125298043094,
      "validation_loss": null
    },
    {
      "epoch": 301,
      "train_loss": 0.003496491195013126,
      "validation_loss": null
    },
    {
      "epoch": 302,
      "train_loss": 0.0038753334536320632,
      "validation_loss": null
    },
    {
      "epoch": 303,
      "train_loss": 0.003470405134268933,
      "validation_loss": null
    },
    {
      "epoch": 304,
      "train_loss": 0.0029987974816726316,
      "validation_loss": null
    },
    {
      "epoch": 305,
      "train_loss": 0.003416892529154817,
      "validation_loss": null
    },
    {
      "epoch": 306,
      "train_loss": 0.003355154896982842,
      "validation_loss": null
    },
    {
      "epoch": 307,
      "train_loss": 0.003556387667130265,
      "validation_loss": null
    },
    {
      "epoch": 308,
      "train_loss": 0.003933262519745363,
      "validation_loss": null
    },
    {
      "epoch": 309,
      "train_loss": 0.004233288009547525,
      "validation_loss": null
    },
    {
      "epoch": 310,
      "train_loss": 0.004236352759309941,
      "validation_loss": null
    },
    {
      "epoch": 311,
      "train_loss": 0.004137200277505649,
      "validation_loss": null
    },
    {
      "epoch": 312,
      "train_loss": 0.00323902516781042,
      "validation_loss": null
    },
    {
      "epoch": 313,
      "train_loss": 0.002820254044814242,
      "validation_loss": null
    },
    {
      "epoch": 314,
      "train_loss": 0.002888422912090189,
      "validation_loss": null
    },
    {
      "epoch": 315,
      "train_loss": 0.0029530402324679825,
      "validation_loss": null
    },
    {
      "epoch": 316,
      "train_loss": 0.0033829915595965255,
      "validation_loss": null
    },
    {
      "epoch": 317,
      "train_loss": 0.002999205090519455,
      "validation_loss": null
    },
    {
      "epoch": 318,
      "train_loss": 0.0031847302646686635,
      "validation_loss": null
    },
    {
      "epoch": 319,
      "train_loss": 0.0027291012617448964,
      "validation_loss": null
    },
    {
      "epoch": 320,
      "train_loss": 0.002542783234578868,
      "validation_loss": null
    },
    {
      "epoch": 321,
      "train_loss": 0.002608803670025534,
      "validation_loss": null
    },
    {
      "epoch": 322,
      "train_loss": 0.00311377773889237,
      "validation_loss": null
    },
    {
      "epoch": 323,
      "train_loss": 0.0026944647801833022,
      "validation_loss": null
    },
    {
      "epoch": 324,
      "train_loss": 0.002959343281367587,
      "validation_loss": null
    },
    {
      "epoch": 325,
      "train_loss": 0.002773700100887153,
      "validation_loss": null
    },
    {
      "epoch": 326,
      "train_loss": 0.0029396458477195767,
      "validation_loss": null
    },
    {
      "epoch": 327,
      "train_loss": 0.0030127096947075594,
      "validation_loss": null
    },
    {
      "epoch": 328,
      "train_loss": 0.0024561648153596455,
      "validation_loss": null
    },
    {
      "epoch": 329,
      "train_loss": 0.002483040391881433,
      "validation_loss": null
    },
    {
      "epoch": 330,
      "train_loss": 0.0024677729880851177,
      "validation_loss": null
    },
    {
      "epoch": 331,
      "train_loss": 0.0026906466292631295,
      "validation_loss": null
    },
    {
      "epoch": 332,
      "train_loss": 0.002538741333410144,
      "validation_loss": null
    },
    {
      "epoch": 333,
      "train_loss": 0.002511067652247018,
      "validation_loss": null
    },
    {
      "epoch": 334,
      "train_loss": 0.0024323934921994805,
      "validation_loss": null
    },
    {
      "epoch": 335,
      "train_loss": 0.002533874694361455,
      "validation_loss": null
    },
    {
      "epoch": 336,
      "train_loss": 0.002739605131662554,
      "validation_loss": null
    },
    {
      "epoch": 337,
      "train_loss": 0.0027496771266063056,
      "validation_loss": null
    },
    {
      "epoch": 338,
      "train_loss": 0.0025661164610129264,
      "validation_loss": null
    },
    {
      "epoch": 339,
      "train_loss": 0.0023675732534482246,
      "validation_loss": null
    },
    {
      "epoch": 340,
      "train_loss": 0.0023215596383023593,
      "validation_loss": null
    },
    {
      "epoch": 341,
      "train_loss": 0.0025091919168415996,
      "validation_loss": null
    },
    {
      "epoch": 342,
      "train_loss": 0.0023871361821268997,
      "validation_loss": null
    },
    {
      "epoch": 343,
      "train_loss": 0.002628644203974141,
      "validation_loss": null
    },
    {
      "epoch": 344,
      "train_loss": 0.0034933185670524836,
      "validation_loss": null
    },
    {
      "epoch": 345,
      "train_loss": 0.0026295893282319107,
      "validation_loss": null
    },
    {
      "epoch": 346,
      "train_loss": 0.0031831515564893684,
      "validation_loss": null
    },
    {
      "epoch": 347,
      "train_loss": 0.003738804953172803,
      "validation_loss": null
    },
    {
      "epoch": 348,
      "train_loss": 0.0032359874134676326,
      "validation_loss": null
    },
    {
      "epoch": 349,
      "train_loss": 0.0038021280197426677,
      "validation_loss": null
    },
    {
      "epoch": 350,
      "train_loss": 0.004926670265073578,
      "validation_loss": null
    },
    {
      "epoch": 351,
      "train_loss": 0.004014301139654385,
      "validation_loss": null
    },
    {
      "epoch": 352,
      "train_loss": 0.0031381711467272705,
      "validation_loss": null
    },
    {
      "epoch": 353,
      "train_loss": 0.004640993274127443,
      "validation_loss": null
    },
    {
      "epoch": 354,
      "train_loss": 0.002831072585346798,
      "validation_loss": null
    },
    {
      "epoch": 355,
      "train_loss": 0.002416776286231147,
      "validation_loss": null
    },
    {
      "epoch": 356,
      "train_loss": 0.0024139160264490377,
      "validation_loss": null
    },
    {
      "epoch": 357,
      "train_loss": 0.0023317571548331114,
      "validation_loss": null
    },
    {
      "epoch": 358,
      "train_loss": 0.0023805076747925747,
      "validation_loss": null
    },
    {
      "epoch": 359,
      "train_loss": 0.002803423739452329,
      "validation_loss": null
    },
    {
      "epoch": 360,
      "train_loss": 0.0026923109673791462,
      "validation_loss": null
    },
    {
      "epoch": 361,
      "train_loss": 0.0024481348615760603,
      "validation_loss": null
    },
    {
      "epoch": 362,
      "train_loss": 0.004293708182457421,
      "validation_loss": null
    },
    {
      "epoch": 363,
      "train_loss": 0.0033260677656572727,
      "validation_loss": null
    },
    {
      "epoch": 364,
      "train_loss": 0.0032970896896181833,
      "validation_loss": null
    },
    {
      "epoch": 365,
      "train_loss": 0.0022455806352405083,
      "validation_loss": null
    },
    {
      "epoch": 366,
      "train_loss": 0.0023721582126907175,
      "validation_loss": null
    },
    {
      "epoch": 367,
      "train_loss": 0.003449244631661309,
      "validation_loss": null
    },
    {
      "epoch": 368,
      "train_loss": 0.0033031703593830266,
      "validation_loss": null
    },
    {
      "epoch": 369,
      "train_loss": 0.0026514387120389277,
      "validation_loss": null
    },
    {
      "epoch": 370,
      "train_loss": 0.0024132754834782747,
      "validation_loss": null
    },
    {
      "epoch": 371,
      "train_loss": 0.0019654477623084355,
      "validation_loss": null
    },
    {
      "epoch": 372,
      "train_loss": 0.002161765044244627,
      "validation_loss": null
    },
    {
      "epoch": 373,
      "train_loss": 0.002295451587997377,
      "validation_loss": null
    },
    {
      "epoch": 374,
      "train_loss": 0.0029018886723659104,
      "validation_loss": null
    },
    {
      "epoch": 375,
      "train_loss": 0.0027353625951541793,
      "validation_loss": null
    },
    {
      "epoch": 376,
      "train_loss": 0.0025056940115367374,
      "validation_loss": null
    },
    {
      "epoch": 377,
      "train_loss": 0.0028722966607246134,
      "validation_loss": null
    },
    {
      "epoch": 378,
      "train_loss": 0.0022272536168909734,
      "validation_loss": null
    },
    {
      "epoch": 379,
      "train_loss": 0.0018800120322137242,
      "validation_loss": null
    },
    {
      "epoch": 380,
      "train_loss": 0.0019591167931341464,
      "validation_loss": null
    },
    {
      "epoch": 381,
      "train_loss": 0.0019300705365215738,
      "validation_loss": null
    },
    {
      "epoch": 382,
      "train_loss": 0.002102218400169578,
      "validation_loss": null
    },
    {
      "epoch": 383,
      "train_loss": 0.0018199052687527405,
      "validation_loss": null
    },
    {
      "epoch": 384,
      "train_loss": 0.0018096970614149338,
      "validation_loss": null
    },
    {
      "epoch": 385,
      "train_loss": 0.0018493630793980425,
      "validation_loss": null
    },
    {
      "epoch": 386,
      "train_loss": 0.001968113408010039,
      "validation_loss": null
    },
    {
      "epoch": 387,
      "train_loss": 0.0018642830869389905,
      "validation_loss": null
    },
    {
      "epoch": 388,
      "train_loss": 0.002697736904438999,
      "validation_loss": null
    },
    {
      "epoch": 389,
      "train_loss": 0.002507087593484256,
      "validation_loss": null
    },
    {
      "epoch": 390,
      "train_loss": 0.0023148400295111868,
      "validation_loss": null
    },
    {
      "epoch": 391,
      "train_loss": 0.002338335500098765,
      "validation_loss": null
    },
    {
      "epoch": 392,
      "train_loss": 0.0021830748818400833,
      "validation_loss": null
    },
    {
      "epoch": 393,
      "train_loss": 0.0032883335111869704,
      "validation_loss": null
    },
    {
      "epoch": 394,
      "train_loss": 0.002497917571518984,
      "validation_loss": null
    },
    {
      "epoch": 395,
      "train_loss": 0.0024762975258959662,
      "validation_loss": null
    },
    {
      "epoch": 396,
      "train_loss": 0.002454788152438899,
      "validation_loss": null
    },
    {
      "epoch": 397,
      "train_loss": 0.002888942033880287,
      "validation_loss": null
    },
    {
      "epoch": 398,
      "train_loss": 0.0026217193679056233,
      "validation_loss": null
    },
    {
      "epoch": 399,
      "train_loss": 0.0020772086766858897,
      "validation_loss": null
    },
    {
      "epoch": 400,
      "train_loss": 0.0019118358484572833,
      "validation_loss": null
    },
    {
      "epoch": 401,
      "train_loss": 0.0015766431525763539,
      "validation_loss": null
    },
    {
      "epoch": 402,
      "train_loss": 0.001682440595080455,
      "validation_loss": null
    },
    {
      "epoch": 403,
      "train_loss": 0.0018034876298366322,
      "validation_loss": null
    },
    {
      "epoch": 404,
      "train_loss": 0.001792813580121017,
      "validation_loss": null
    },
    {
      "epoch": 405,
      "train_loss": 0.0016782033425341877,
      "validation_loss": null
    },
    {
      "epoch": 406,
      "train_loss": 0.0016994940362767214,
      "validation_loss": null
    },
    {
      "epoch": 407,
      "train_loss": 0.001675931607476539,
      "validation_loss": null
    },
    {
      "epoch": 408,
      "train_loss": 0.0017266321616868179,
      "validation_loss": null
    },
    {
      "epoch": 409,
      "train_loss": 0.0016121496446430683,
      "validation_loss": null
    },
    {
      "epoch": 410,
      "train_loss": 0.002222938579507172,
      "validation_loss": null
    },
    {
      "epoch": 411,
      "train_loss": 0.002008281130757597,
      "validation_loss": null
    },
    {
      "epoch": 412,
      "train_loss": 0.0023834886313933465,
      "validation_loss": null
    },
    {
      "epoch": 413,
      "train_loss": 0.0021342152289839256,
      "validation_loss": null
    },
    {
      "epoch": 414,
      "train_loss": 0.0033111447701230645,
      "validation_loss": null
    },
    {
      "epoch": 415,
      "train_loss": 0.0023776069671536484,
      "validation_loss": null
    },
    {
      "epoch": 416,
      "train_loss": 0.001799002115149051,
      "validation_loss": null
    },
    {
      "epoch": 417,
      "train_loss": 0.0019795580042733085,
      "validation_loss": null
    },
    {
      "epoch": 418,
      "train_loss": 0.0018493153620511293,
      "validation_loss": null
    },
    {
      "epoch": 419,
      "train_loss": 0.0015418068505823612,
      "validation_loss": null
    },
    {
      "epoch": 420,
      "train_loss": 0.0016077576308614677,
      "validation_loss": null
    },
    {
      "epoch": 421,
      "train_loss": 0.001624641357921064,
      "validation_loss": null
    },
    {
      "epoch": 422,
      "train_loss": 0.002070068698635118,
      "validation_loss": null
    },
    {
      "epoch": 423,
      "train_loss": 0.0016266183358513648,
      "validation_loss": null
    },
    {
      "epoch": 424,
      "train_loss": 0.0015054367055806022,
      "validation_loss": null
    },
    {
      "epoch": 425,
      "train_loss": 0.0017683562667419512,
      "validation_loss": null
    },
    {
      "epoch": 426,
      "train_loss": 0.0017313421703875065,
      "validation_loss": null
    },
    {
      "epoch": 427,
      "train_loss": 0.0017714424757286906,
      "validation_loss": null
    },
    {
      "epoch": 428,
      "train_loss": 0.0018054613998780649,
      "validation_loss": null
    },
    {
      "epoch": 429,
      "train_loss": 0.0017225874988879594,
      "validation_loss": null
    },
    {
      "epoch": 430,
      "train_loss": 0.0014720905908486908,
      "validation_loss": null
    },
    {
      "epoch": 431,
      "train_loss": 0.002346621994446549,
      "validation_loss": null
    },
    {
      "epoch": 432,
      "train_loss": 0.0015303119111599193,
      "validation_loss": null
    },
    {
      "epoch": 433,
      "train_loss": 0.0014910612048374282,
      "validation_loss": null
    },
    {
      "epoch": 434,
      "train_loss": 0.0018138738167989585,
      "validation_loss": null
    },
    {
      "epoch": 435,
      "train_loss": 0.0016299704309656387,
      "validation_loss": null
    },
    {
      "epoch": 436,
      "train_loss": 0.0014369298458202845,
      "validation_loss": null
    },
    {
      "epoch": 437,
      "train_loss": 0.0015401163901616302,
      "validation_loss": null
    },
    {
      "epoch": 438,
      "train_loss": 0.0015378450819601615,
      "validation_loss": null
    },
    {
      "epoch": 439,
      "train_loss": 0.0017611317905700868,
      "validation_loss": null
    },
    {
      "epoch": 440,
      "train_loss": 0.0017407671548426151,
      "validation_loss": null
    },
    {
      "epoch": 441,
      "train_loss": 0.00142780765114973,
      "validation_loss": null
    },
    {
      "epoch": 442,
      "train_loss": 0.0013312769329382314,
      "validation_loss": null
    },
    {
      "epoch": 443,
      "train_loss": 0.0013055020762193534,
      "validation_loss": null
    },
    {
      "epoch": 444,
      "train_loss": 0.0023366522380254334,
      "validation_loss": null
    },
    {
      "epoch": 445,
      "train_loss": 0.002343782805837691,
      "validation_loss": null
    },
    {
      "epoch": 446,
      "train_loss": 0.001710666007258826,
      "validation_loss": null
    },
    {
      "epoch": 447,
      "train_loss": 0.0019514495765583382,
      "validation_loss": null
    },
    {
      "epoch": 448,
      "train_loss": 0.0023734846941402387,
      "validation_loss": null
    },
    {
      "epoch": 449,
      "train_loss": 0.002406020752257771,
      "validation_loss": null
    },
    {
      "epoch": 450,
      "train_loss": 0.0033150811731401417,
      "validation_loss": null
    },
    {
      "epoch": 451,
      "train_loss": 0.002069740452700191,
      "validation_loss": null
    },
    {
      "epoch": 452,
      "train_loss": 0.0016882496096918152,
      "validation_loss": null
    },
    {
      "epoch": 453,
      "train_loss": 0.001348444713383085,
      "validation_loss": null
    },
    {
      "epoch": 454,
      "train_loss": 0.001370382430549297,
      "validation_loss": null
    },
    {
      "epoch": 455,
      "train_loss": 0.0014399226121087042,
      "validation_loss": null
    },
    {
      "epoch": 456,
      "train_loss": 0.0013240136775291627,
      "validation_loss": null
    },
    {
      "epoch": 457,
      "train_loss": 0.0016772119884586169,
      "validation_loss": null
    },
    {
      "epoch": 458,
      "train_loss": 0.002497129750231074,
      "validation_loss": null
    },
    {
      "epoch": 459,
      "train_loss": 0.004887171956296597,
      "validation_loss": null
    },
    {
      "epoch": 460,
      "train_loss": 0.002787973279029959,
      "validation_loss": null
    },
    {
      "epoch": 461,
      "train_loss": 0.0016718226510824428,
      "validation_loss": null
    },
    {
      "epoch": 462,
      "train_loss": 0.0014223844304473863,
      "validation_loss": null
    },
    {
      "epoch": 463,
      "train_loss": 0.0012235803038088812,
      "validation_loss": null
    },
    {
      "epoch": 464,
      "train_loss": 0.0016975252591590914,
      "validation_loss": null
    },
    {
      "epoch": 465,
      "train_loss": 0.0016089150190560354,
      "validation_loss": null
    },
    {
      "epoch": 466,
      "train_loss": 0.0012468116150961982,
      "validation_loss": null
    },
    {
      "epoch": 467,
      "train_loss": 0.0013304397580213845,
      "validation_loss": null
    },
    {
      "epoch": 468,
      "train_loss": 0.0016677537932991982,
      "validation_loss": null
    },
    {
      "epoch": 469,
      "train_loss": 0.001457325894282096,
      "validation_loss": null
    },
    {
      "epoch": 470,
      "train_loss": 0.0014362211804836988,
      "validation_loss": null
    },
    {
      "epoch": 471,
      "train_loss": 0.0012135182365050747,
      "validation_loss": null
    },
    {
      "epoch": 472,
      "train_loss": 0.0011879722135038013,
      "validation_loss": null
    },
    {
      "epoch": 473,
      "train_loss": 0.0012981936775354876,
      "validation_loss": null
    },
    {
      "epoch": 474,
      "train_loss": 0.0014243280101153585,
      "validation_loss": null
    },
    {
      "epoch": 475,
      "train_loss": 0.0012735012763490279,
      "validation_loss": null
    },
    {
      "epoch": 476,
      "train_loss": 0.001338638274723457,
      "validation_loss": null
    },
    {
      "epoch": 477,
      "train_loss": 0.0012659797535484864,
      "validation_loss": null
    },
    {
      "epoch": 478,
      "train_loss": 0.0012727871200897628,
      "validation_loss": null
    },
    {
      "epoch": 479,
      "train_loss": 0.0010557048307317826,
      "validation_loss": null
    },
    {
      "epoch": 480,
      "train_loss": 0.0010748419574358398,
      "validation_loss": null
    },
    {
      "epoch": 481,
      "train_loss": 0.0010628142238905032,
      "validation_loss": null
    },
    {
      "epoch": 482,
      "train_loss": 0.0014403381048598224,
      "validation_loss": null
    },
    {
      "epoch": 483,
      "train_loss": 0.0028155375944657456,
      "validation_loss": null
    },
    {
      "epoch": 484,
      "train_loss": 0.0038840800277992254,
      "validation_loss": null
    },
    {
      "epoch": 485,
      "train_loss": 0.002181686797282762,
      "validation_loss": null
    },
    {
      "epoch": 486,
      "train_loss": 0.0016098305156144004,
      "validation_loss": null
    },
    {
      "epoch": 487,
      "train_loss": 0.0013730773547043402,
      "validation_loss": null
    },
    {
      "epoch": 488,
      "train_loss": 0.0011189136938709351,
      "validation_loss": null
    },
    {
      "epoch": 489,
      "train_loss": 0.0010944683065948386,
      "validation_loss": null
    },
    {
      "epoch": 490,
      "train_loss": 0.0013651933081241117,
      "validation_loss": null
    },
    {
      "epoch": 491,
      "train_loss": 0.0010147948826973636,
      "validation_loss": null
    },
    {
      "epoch": 492,
      "train_loss": 0.001021602954198089,
      "validation_loss": null
    },
    {
      "epoch": 493,
      "train_loss": 0.0010278864323885904,
      "validation_loss": null
    },
    {
      "epoch": 494,
      "train_loss": 0.0010955191424323453,
      "validation_loss": null
    },
    {
      "epoch": 495,
      "train_loss": 0.0016246529024404783,
      "validation_loss": null
    },
    {
      "epoch": 496,
      "train_loss": 0.0015575338620692492,
      "validation_loss": null
    },
    {
      "epoch": 497,
      "train_loss": 0.0012474947402046786,
      "validation_loss": null
    },
    {
      "epoch": 498,
      "train_loss": 0.0012725245517989,
      "validation_loss": null
    },
    {
      "epoch": 499,
      "train_loss": 0.0014127154330102105,
      "validation_loss": null
    },
    {
      "epoch": 500,
      "train_loss": 0.0011864390754554835,
      "validation_loss": null
    },
    {
      "epoch": 501,
      "train_loss": 0.001129438020547645,
      "validation_loss": null
    },
    {
      "epoch": 502,
      "train_loss": 0.001250871903418253,
      "validation_loss": null
    },
    {
      "epoch": 503,
      "train_loss": 0.0013569826906960872,
      "validation_loss": null
    },
    {
      "epoch": 504,
      "train_loss": 0.0018775159001557364,
      "validation_loss": null
    },
    {
      "epoch": 505,
      "train_loss": 0.0013590313610620797,
      "validation_loss": null
    },
    {
      "epoch": 506,
      "train_loss": 0.0011621166615643436,
      "validation_loss": null
    },
    {
      "epoch": 507,
      "train_loss": 0.001267249055672437,
      "validation_loss": null
    },
    {
      "epoch": 508,
      "train_loss": 0.0012569766194145712,
      "validation_loss": null
    },
    {
      "epoch": 509,
      "train_loss": 0.0010731216494200958,
      "validation_loss": null
    },
    {
      "epoch": 510,
      "train_loss": 0.0009769175085239112,
      "validation_loss": null
    },
    {
      "epoch": 511,
      "train_loss": 0.0010710766137991515,
      "validation_loss": null
    },
    {
      "epoch": 512,
      "train_loss": 0.00102624125752805,
      "validation_loss": null
    },
    {
      "epoch": 513,
      "train_loss": 0.0009344021608639094,
      "validation_loss": null
    },
    {
      "epoch": 514,
      "train_loss": 0.0008649438128082289,
      "validation_loss": null
    },
    {
      "epoch": 515,
      "train_loss": 0.0008566991244960162,
      "validation_loss": null
    },
    {
      "epoch": 516,
      "train_loss": 0.0008855753565310604,
      "validation_loss": null
    },
    {
      "epoch": 517,
      "train_loss": 0.0012585325910347616,
      "validation_loss": null
    },
    {
      "epoch": 518,
      "train_loss": 0.0010919167155710359,
      "validation_loss": null
    },
    {
      "epoch": 519,
      "train_loss": 0.0010329719128397603,
      "validation_loss": null
    },
    {
      "epoch": 520,
      "train_loss": 0.0011531509614239137,
      "validation_loss": null
    },
    {
      "epoch": 521,
      "train_loss": 0.0008810809393051184,
      "validation_loss": null
    },
    {
      "epoch": 522,
      "train_loss": 0.001099533204170358,
      "validation_loss": null
    },
    {
      "epoch": 523,
      "train_loss": 0.001034990415468605,
      "validation_loss": null
    },
    {
      "epoch": 524,
      "train_loss": 0.0008648545416589412,
      "validation_loss": null
    },
    {
      "epoch": 525,
      "train_loss": 0.0011056426899611121,
      "validation_loss": null
    },
    {
      "epoch": 526,
      "train_loss": 0.0014215051390541096,
      "validation_loss": null
    },
    {
      "epoch": 527,
      "train_loss": 0.001028605284065836,
      "validation_loss": null
    },
    {
      "epoch": 528,
      "train_loss": 0.0011519974537400736,
      "validation_loss": null
    },
    {
      "epoch": 529,
      "train_loss": 0.001031434305736588,
      "validation_loss": null
    },
    {
      "epoch": 530,
      "train_loss": 0.001098669936052627,
      "validation_loss": null
    },
    {
      "epoch": 531,
      "train_loss": 0.0015457833651453257,
      "validation_loss": null
    },
    {
      "epoch": 532,
      "train_loss": 0.0035031333229401046,
      "validation_loss": null
    },
    {
      "epoch": 533,
      "train_loss": 0.004747819310675065,
      "validation_loss": null
    },
    {
      "epoch": 534,
      "train_loss": 0.0021403993143596584,
      "validation_loss": null
    },
    {
      "epoch": 535,
      "train_loss": 0.001592750143673685,
      "validation_loss": null
    },
    {
      "epoch": 536,
      "train_loss": 0.0012327268745543228,
      "validation_loss": null
    },
    {
      "epoch": 537,
      "train_loss": 0.0008641941951484316,
      "validation_loss": null
    },
    {
      "epoch": 538,
      "train_loss": 0.0009017998947658472,
      "validation_loss": null
    },
    {
      "epoch": 539,
      "train_loss": 0.0011069035422729535,
      "validation_loss": null
    },
    {
      "epoch": 540,
      "train_loss": 0.0012940537093931602,
      "validation_loss": null
    },
    {
      "epoch": 541,
      "train_loss": 0.0010821615707957083,
      "validation_loss": null
    },
    {
      "epoch": 542,
      "train_loss": 0.0013859286037687627,
      "validation_loss": null
    },
    {
      "epoch": 543,
      "train_loss": 0.0012072272753963869,
      "validation_loss": null
    },
    {
      "epoch": 544,
      "train_loss": 0.0010089589244065185,
      "validation_loss": null
    },
    {
      "epoch": 545,
      "train_loss": 0.0007356009769460394,
      "validation_loss": null
    },
    {
      "epoch": 546,
      "train_loss": 0.0006832077859952631,
      "validation_loss": null
    },
    {
      "epoch": 547,
      "train_loss": 0.0007029162758650879,
      "validation_loss": null
    },
    {
      "epoch": 548,
      "train_loss": 0.0009739138962080082,
      "validation_loss": null
    },
    {
      "epoch": 549,
      "train_loss": 0.0008092718780972064,
      "validation_loss": null
    },
    {
      "epoch": 550,
      "train_loss": 0.0008525917688125952,
      "validation_loss": null
    },
    {
      "epoch": 551,
      "train_loss": 0.0009404566436488596,
      "validation_loss": null
    },
    {
      "epoch": 552,
      "train_loss": 0.001248986706034177,
      "validation_loss": null
    },
    {
      "epoch": 553,
      "train_loss": 0.0009006085197648241,
      "validation_loss": null
    },
    {
      "epoch": 554,
      "train_loss": 0.0007988720414383957,
      "validation_loss": null
    },
    {
      "epoch": 555,
      "train_loss": 0.0008041248462783793,
      "validation_loss": null
    },
    {
      "epoch": 556,
      "train_loss": 0.0009490093539675905,
      "validation_loss": null
    },
    {
      "epoch": 557,
      "train_loss": 0.0008660104487919145,
      "validation_loss": null
    },
    {
      "epoch": 558,
      "train_loss": 0.0009748926838963396,
      "validation_loss": null
    },
    {
      "epoch": 559,
      "train_loss": 0.0010097170921249522,
      "validation_loss": null
    },
    {
      "epoch": 560,
      "train_loss": 0.0011272955258997779,
      "validation_loss": null
    },
    {
      "epoch": 561,
      "train_loss": 0.0008235727889566786,
      "validation_loss": null
    },
    {
      "epoch": 562,
      "train_loss": 0.000829009513836354,
      "validation_loss": null
    },
    {
      "epoch": 563,
      "train_loss": 0.0005986917628130565,
      "validation_loss": null
    },
    {
      "epoch": 564,
      "train_loss": 0.0006338227831292897,
      "validation_loss": null
    },
    {
      "epoch": 565,
      "train_loss": 0.0007160002682616727,
      "validation_loss": null
    },
    {
      "epoch": 566,
      "train_loss": 0.0011153416837462122,
      "validation_loss": null
    },
    {
      "epoch": 567,
      "train_loss": 0.0009976281241203349,
      "validation_loss": null
    },
    {
      "epoch": 568,
      "train_loss": 0.0009224288320789734,
      "validation_loss": null
    },
    {
      "epoch": 569,
      "train_loss": 0.0011628330879223843,
      "validation_loss": null
    },
    {
      "epoch": 570,
      "train_loss": 0.0011371307191438973,
      "validation_loss": null
    },
    {
      "epoch": 571,
      "train_loss": 0.0013258387140619259,
      "validation_loss": null
    },
    {
      "epoch": 572,
      "train_loss": 0.0010498074698261917,
      "validation_loss": null
    },
    {
      "epoch": 573,
      "train_loss": 0.0011422813857077723,
      "validation_loss": null
    },
    {
      "epoch": 574,
      "train_loss": 0.001340485591855314,
      "validation_loss": null
    },
    {
      "epoch": 575,
      "train_loss": 0.0014870192260584897,
      "validation_loss": null
    },
    {
      "epoch": 576,
      "train_loss": 0.0012877693128151198,
      "validation_loss": null
    },
    {
      "epoch": 577,
      "train_loss": 0.0017983641979905467,
      "validation_loss": null
    },
    {
      "epoch": 578,
      "train_loss": 0.0017484428050617378,
      "validation_loss": null
    },
    {
      "epoch": 579,
      "train_loss": 0.0019090913766477671,
      "validation_loss": null
    },
    {
      "epoch": 580,
      "train_loss": 0.002623699307959113,
      "validation_loss": null
    },
    {
      "epoch": 581,
      "train_loss": 0.001636670882968853,
      "validation_loss": null
    },
    {
      "epoch": 582,
      "train_loss": 0.0015738988424547845,
      "validation_loss": null
    },
    {
      "epoch": 583,
      "train_loss": 0.0013038454732547204,
      "validation_loss": null
    },
    {
      "epoch": 584,
      "train_loss": 0.0008648128067660662,
      "validation_loss": null
    },
    {
      "epoch": 585,
      "train_loss": 0.0007265049466190652,
      "validation_loss": null
    },
    {
      "epoch": 586,
      "train_loss": 0.0006339751125779003,
      "validation_loss": null
    },
    {
      "epoch": 587,
      "train_loss": 0.0007013728091907171,
      "validation_loss": null
    },
    {
      "epoch": 588,
      "train_loss": 0.0008139941104066869,
      "validation_loss": null
    },
    {
      "epoch": 589,
      "train_loss": 0.0006988284779557338,
      "validation_loss": null
    },
    {
      "epoch": 590,
      "train_loss": 0.0008702194285837726,
      "validation_loss": null
    },
    {
      "epoch": 591,
      "train_loss": 0.0007818564158191697,
      "validation_loss": null
    },
    {
      "epoch": 592,
      "train_loss": 0.000605029775114316,
      "validation_loss": null
    },
    {
      "epoch": 593,
      "train_loss": 0.0005654134017984486,
      "validation_loss": null
    },
    {
      "epoch": 594,
      "train_loss": 0.0007797512355157071,
      "validation_loss": null
    },
    {
      "epoch": 595,
      "train_loss": 0.0018745364440191123,
      "validation_loss": null
    },
    {
      "epoch": 596,
      "train_loss": 0.001333580429976185,
      "validation_loss": null
    },
    {
      "epoch": 597,
      "train_loss": 0.0012900693173934189,
      "validation_loss": null
    },
    {
      "epoch": 598,
      "train_loss": 0.0009196675059178637,
      "validation_loss": null
    },
    {
      "epoch": 599,
      "train_loss": 0.0008083565626293421,
      "validation_loss": null
    },
    {
      "epoch": 600,
      "train_loss": 0.000589578616200015,
      "validation_loss": null
    },
    {
      "epoch": 601,
      "train_loss": 0.0004704098812200957,
      "validation_loss": null
    },
    {
      "epoch": 602,
      "train_loss": 0.0008606650032258282,
      "validation_loss": null
    },
    {
      "epoch": 603,
      "train_loss": 0.0009774430881306115,
      "validation_loss": null
    },
    {
      "epoch": 604,
      "train_loss": 0.001064982510999673,
      "validation_loss": null
    },
    {
      "epoch": 605,
      "train_loss": 0.0012885252718761978,
      "validation_loss": null
    },
    {
      "epoch": 606,
      "train_loss": 0.0010566361274363266,
      "validation_loss": null
    },
    {
      "epoch": 607,
      "train_loss": 0.0012153556906721657,
      "validation_loss": null
    },
    {
      "epoch": 608,
      "train_loss": 0.0011917283473950294,
      "validation_loss": null
    },
    {
      "epoch": 609,
      "train_loss": 0.001178099773824215,
      "validation_loss": null
    },
    {
      "epoch": 610,
      "train_loss": 0.0012450777964356046,
      "validation_loss": null
    },
    {
      "epoch": 611,
      "train_loss": 0.001355303541964127,
      "validation_loss": null
    },
    {
      "epoch": 612,
      "train_loss": 0.0008080563792544934,
      "validation_loss": null
    },
    {
      "epoch": 613,
      "train_loss": 0.0007647440393662287,
      "validation_loss": null
    },
    {
      "epoch": 614,
      "train_loss": 0.0006677571250798388,
      "validation_loss": null
    },
    {
      "epoch": 615,
      "train_loss": 0.0004772229876834899,
      "validation_loss": null
    },
    {
      "epoch": 616,
      "train_loss": 0.0004774350317246798,
      "validation_loss": null
    },
    {
      "epoch": 617,
      "train_loss": 0.0005703315449257692,
      "validation_loss": null
    },
    {
      "epoch": 618,
      "train_loss": 0.000603878514892939,
      "validation_loss": null
    },
    {
      "epoch": 619,
      "train_loss": 0.000798118915150149,
      "validation_loss": null
    },
    {
      "epoch": 620,
      "train_loss": 0.0006523270114687168,
      "validation_loss": null
    },
    {
      "epoch": 621,
      "train_loss": 0.0005437032474825779,
      "validation_loss": null
    },
    {
      "epoch": 622,
      "train_loss": 0.0005661764751291937,
      "validation_loss": null
    },
    {
      "epoch": 623,
      "train_loss": 0.0009124526598801216,
      "validation_loss": null
    },
    {
      "epoch": 624,
      "train_loss": 0.0009826274342938429,
      "validation_loss": null
    },
    {
      "epoch": 625,
      "train_loss": 0.0008503595147178405,
      "validation_loss": null
    },
    {
      "epoch": 626,
      "train_loss": 0.0007981223623371787,
      "validation_loss": null
    },
    {
      "epoch": 627,
      "train_loss": 0.0008182048752334797,
      "validation_loss": null
    },
    {
      "epoch": 628,
      "train_loss": 0.0007195830241673522,
      "validation_loss": null
    },
    {
      "epoch": 629,
      "train_loss": 0.0005258279464518031,
      "validation_loss": null
    },
    {
      "epoch": 630,
      "train_loss": 0.0007059997005853802,
      "validation_loss": null
    },
    {
      "epoch": 631,
      "train_loss": 0.0006042578994917373,
      "validation_loss": null
    },
    {
      "epoch": 632,
      "train_loss": 0.0005830095324199647,
      "validation_loss": null
    },
    {
      "epoch": 633,
      "train_loss": 0.0010627605822972125,
      "validation_loss": null
    },
    {
      "epoch": 634,
      "train_loss": 0.0029911388410255313,
      "validation_loss": null
    },
    {
      "epoch": 635,
      "train_loss": 0.004620152978330023,
      "validation_loss": null
    },
    {
      "epoch": 636,
      "train_loss": 0.004028342534891433,
      "validation_loss": null
    },
    {
      "epoch": 637,
      "train_loss": 0.0024298828405638537,
      "validation_loss": null
    },
    {
      "epoch": 638,
      "train_loss": 0.0011102078650664124,
      "validation_loss": null
    },
    {
      "epoch": 639,
      "train_loss": 0.0008798236699981822,
      "validation_loss": null
    },
    {
      "epoch": 640,
      "train_loss": 0.001749213535933652,
      "validation_loss": null
    },
    {
      "epoch": 641,
      "train_loss": 0.0015337064461265174,
      "validation_loss": null
    },
    {
      "epoch": 642,
      "train_loss": 0.0013935955422413018,
      "validation_loss": null
    },
    {
      "epoch": 643,
      "train_loss": 0.0017164308616581063,
      "validation_loss": null
    },
    {
      "epoch": 644,
      "train_loss": 0.0014868262805975974,
      "validation_loss": null
    },
    {
      "epoch": 645,
      "train_loss": 0.0013385932672665352,
      "validation_loss": null
    },
    {
      "epoch": 646,
      "train_loss": 0.001249328223315792,
      "validation_loss": null
    },
    {
      "epoch": 647,
      "train_loss": 0.0015059268011504577,
      "validation_loss": null
    },
    {
      "epoch": 648,
      "train_loss": 0.001004288173539357,
      "validation_loss": null
    },
    {
      "epoch": 649,
      "train_loss": 0.0009359037244899406,
      "validation_loss": null
    },
    {
      "epoch": 650,
      "train_loss": 0.0007127317431796757,
      "validation_loss": null
    },
    {
      "epoch": 651,
      "train_loss": 0.0006739499319034318,
      "validation_loss": null
    },
    {
      "epoch": 652,
      "train_loss": 0.000616415963223618,
      "validation_loss": null
    },
    {
      "epoch": 653,
      "train_loss": 0.0005735752638429403,
      "validation_loss": null
    },
    {
      "epoch": 654,
      "train_loss": 0.0005594298159444912,
      "validation_loss": null
    },
    {
      "epoch": 655,
      "train_loss": 0.0007224194155747278,
      "validation_loss": null
    },
    {
      "epoch": 656,
      "train_loss": 0.0006066749305723028,
      "validation_loss": null
    },
    {
      "epoch": 657,
      "train_loss": 0.0004903447439169718,
      "validation_loss": null
    },
    {
      "epoch": 658,
      "train_loss": 0.0004226708260830492,
      "validation_loss": null
    },
    {
      "epoch": 659,
      "train_loss": 0.0003772541013960209,
      "validation_loss": null
    },
    {
      "epoch": 660,
      "train_loss": 0.00037945607942270324,
      "validation_loss": null
    },
    {
      "epoch": 661,
      "train_loss": 0.0004271870776493516,
      "validation_loss": null
    },
    {
      "epoch": 662,
      "train_loss": 0.00047619447448394366,
      "validation_loss": null
    },
    {
      "epoch": 663,
      "train_loss": 0.00047110634558420215,
      "validation_loss": null
    },
    {
      "epoch": 664,
      "train_loss": 0.0006045650839546903,
      "validation_loss": null
    },
    {
      "epoch": 665,
      "train_loss": 0.0003875274754439791,
      "validation_loss": null
    },
    {
      "epoch": 666,
      "train_loss": 0.0004559218003931973,
      "validation_loss": null
    },
    {
      "epoch": 667,
      "train_loss": 0.000533404414050488,
      "validation_loss": null
    },
    {
      "epoch": 668,
      "train_loss": 0.00047571659605536197,
      "validation_loss": null
    },
    {
      "epoch": 669,
      "train_loss": 0.0003712218701063345,
      "validation_loss": null
    },
    {
      "epoch": 670,
      "train_loss": 0.0003048747409290324,
      "validation_loss": null
    },
    {
      "epoch": 671,
      "train_loss": 0.00040234804974817153,
      "validation_loss": null
    },
    {
      "epoch": 672,
      "train_loss": 0.0003424733020235888,
      "validation_loss": null
    },
    {
      "epoch": 673,
      "train_loss": 0.0003490455565042794,
      "validation_loss": null
    },
    {
      "epoch": 674,
      "train_loss": 0.00042662710054881044,
      "validation_loss": null
    },
    {
      "epoch": 675,
      "train_loss": 0.0004778123725878282,
      "validation_loss": null
    },
    {
      "epoch": 676,
      "train_loss": 0.00037435190299422375,
      "validation_loss": null
    },
    {
      "epoch": 677,
      "train_loss": 0.00033847030620866764,
      "validation_loss": null
    },
    {
      "epoch": 678,
      "train_loss": 0.0003717024956131354,
      "validation_loss": null
    },
    {
      "epoch": 679,
      "train_loss": 0.0005497620570369893,
      "validation_loss": null
    },
    {
      "epoch": 680,
      "train_loss": 0.0005191572967709766,
      "validation_loss": null
    },
    {
      "epoch": 681,
      "train_loss": 0.0003660461289756414,
      "validation_loss": null
    },
    {
      "epoch": 682,
      "train_loss": 0.00031037702703745,
      "validation_loss": null
    },
    {
      "epoch": 683,
      "train_loss": 0.0003782889656981246,
      "validation_loss": null
    },
    {
      "epoch": 684,
      "train_loss": 0.00029880774673074484,
      "validation_loss": null
    },
    {
      "epoch": 685,
      "train_loss": 0.0003162014359582423,
      "validation_loss": null
    },
    {
      "epoch": 686,
      "train_loss": 0.00034273682912397716,
      "validation_loss": null
    },
    {
      "epoch": 687,
      "train_loss": 0.0003832030181203865,
      "validation_loss": null
    },
    {
      "epoch": 688,
      "train_loss": 0.0004528703187437107,
      "validation_loss": null
    },
    {
      "epoch": 689,
      "train_loss": 0.0004965349022919933,
      "validation_loss": null
    },
    {
      "epoch": 690,
      "train_loss": 0.0003796764697957163,
      "validation_loss": null
    },
    {
      "epoch": 691,
      "train_loss": 0.00043779030157667067,
      "validation_loss": null
    },
    {
      "epoch": 692,
      "train_loss": 0.00045911876238986023,
      "validation_loss": null
    },
    {
      "epoch": 693,
      "train_loss": 0.00039308944542426616,
      "validation_loss": null
    },
    {
      "epoch": 694,
      "train_loss": 0.00037132181266012293,
      "validation_loss": null
    },
    {
      "epoch": 695,
      "train_loss": 0.0003367499416021423,
      "validation_loss": null
    },
    {
      "epoch": 696,
      "train_loss": 0.00033475773913475376,
      "validation_loss": null
    },
    {
      "epoch": 697,
      "train_loss": 0.00033448587702069845,
      "validation_loss": null
    },
    {
      "epoch": 698,
      "train_loss": 0.00036255762420801655,
      "validation_loss": null
    },
    {
      "epoch": 699,
      "train_loss": 0.00033100313804526295,
      "validation_loss": null
    },
    {
      "epoch": 700,
      "train_loss": 0.00035446430653488886,
      "validation_loss": null
    },
    {
      "epoch": 701,
      "train_loss": 0.00036277274032019905,
      "validation_loss": null
    },
    {
      "epoch": 702,
      "train_loss": 0.0004895062722627901,
      "validation_loss": null
    },
    {
      "epoch": 703,
      "train_loss": 0.0003331922392438476,
      "validation_loss": null
    },
    {
      "epoch": 704,
      "train_loss": 0.00034646163274171867,
      "validation_loss": null
    },
    {
      "epoch": 705,
      "train_loss": 0.000532367687103235,
      "validation_loss": null
    },
    {
      "epoch": 706,
      "train_loss": 0.0008875328235767989,
      "validation_loss": null
    },
    {
      "epoch": 707,
      "train_loss": 0.0005786652618553489,
      "validation_loss": null
    },
    {
      "epoch": 708,
      "train_loss": 0.0004936897765017218,
      "validation_loss": null
    },
    {
      "epoch": 709,
      "train_loss": 0.0006375221904211988,
      "validation_loss": null
    },
    {
      "epoch": 710,
      "train_loss": 0.0008634130095338656,
      "validation_loss": null
    },
    {
      "epoch": 711,
      "train_loss": 0.001180687194897069,
      "validation_loss": null
    },
    {
      "epoch": 712,
      "train_loss": 0.0016682428799362646,
      "validation_loss": null
    },
    {
      "epoch": 713,
      "train_loss": 0.0011179932076225264,
      "validation_loss": null
    },
    {
      "epoch": 714,
      "train_loss": 0.0005213383345916453,
      "validation_loss": null
    },
    {
      "epoch": 715,
      "train_loss": 0.0004341808542247034,
      "validation_loss": null
    },
    {
      "epoch": 716,
      "train_loss": 0.00036909558144139335,
      "validation_loss": null
    },
    {
      "epoch": 717,
      "train_loss": 0.0004946183633162744,
      "validation_loss": null
    },
    {
      "epoch": 718,
      "train_loss": 0.0008779161855474942,
      "validation_loss": null
    },
    {
      "epoch": 719,
      "train_loss": 0.0012334007252421644,
      "validation_loss": null
    },
    {
      "epoch": 720,
      "train_loss": 0.002001119615872287,
      "validation_loss": null
    },
    {
      "epoch": 721,
      "train_loss": 0.0008231558798191448,
      "validation_loss": null
    },
    {
      "epoch": 722,
      "train_loss": 0.00044511891772142716,
      "validation_loss": null
    },
    {
      "epoch": 723,
      "train_loss": 0.0005624747330633303,
      "validation_loss": null
    },
    {
      "epoch": 724,
      "train_loss": 0.0008221717046884199,
      "validation_loss": null
    },
    {
      "epoch": 725,
      "train_loss": 0.0009721621851592014,
      "validation_loss": null
    },
    {
      "epoch": 726,
      "train_loss": 0.00040295421462764754,
      "validation_loss": null
    },
    {
      "epoch": 727,
      "train_loss": 0.00036459328516179486,
      "validation_loss": null
    },
    {
      "epoch": 728,
      "train_loss": 0.0004398418920269857,
      "validation_loss": null
    },
    {
      "epoch": 729,
      "train_loss": 0.00043838209239766,
      "validation_loss": null
    },
    {
      "epoch": 730,
      "train_loss": 0.0013029819496700333,
      "validation_loss": null
    },
    {
      "epoch": 731,
      "train_loss": 0.0022143039774770537,
      "validation_loss": null
    },
    {
      "epoch": 732,
      "train_loss": 0.003254626890540951,
      "validation_loss": null
    },
    {
      "epoch": 733,
      "train_loss": 0.003085822319715387,
      "validation_loss": null
    },
    {
      "epoch": 734,
      "train_loss": 0.0018571344618168143,
      "validation_loss": null
    },
    {
      "epoch": 735,
      "train_loss": 0.0018794608768075705,
      "validation_loss": null
    },
    {
      "epoch": 736,
      "train_loss": 0.003132524337464323,
      "validation_loss": null
    },
    {
      "epoch": 737,
      "train_loss": 0.001637473805911011,
      "validation_loss": null
    },
    {
      "epoch": 738,
      "train_loss": 0.0014372434492947327,
      "validation_loss": null
    },
    {
      "epoch": 739,
      "train_loss": 0.0011355442693457007,
      "validation_loss": null
    },
    {
      "epoch": 740,
      "train_loss": 0.0007525967246490634,
      "validation_loss": null
    },
    {
      "epoch": 741,
      "train_loss": 0.0006051952271567037,
      "validation_loss": null
    },
    {
      "epoch": 742,
      "train_loss": 0.0003967844822909683,
      "validation_loss": null
    },
    {
      "epoch": 743,
      "train_loss": 0.0003439404002468412,
      "validation_loss": null
    },
    {
      "epoch": 744,
      "train_loss": 0.0003556681589947806,
      "validation_loss": null
    },
    {
      "epoch": 745,
      "train_loss": 0.0003045500391939034,
      "validation_loss": null
    },
    {
      "epoch": 746,
      "train_loss": 0.000289288461014318,
      "validation_loss": null
    },
    {
      "epoch": 747,
      "train_loss": 0.0006303900251320252,
      "validation_loss": null
    },
    {
      "epoch": 748,
      "train_loss": 0.0005600673547531995,
      "validation_loss": null
    },
    {
      "epoch": 749,
      "train_loss": 0.0005027666532744964,
      "validation_loss": null
    },
    {
      "epoch": 750,
      "train_loss": 0.0004092266414469729,
      "validation_loss": null
    },
    {
      "epoch": 751,
      "train_loss": 0.00039953314343519096,
      "validation_loss": null
    },
    {
      "epoch": 752,
      "train_loss": 0.0005716280364948842,
      "validation_loss": null
    },
    {
      "epoch": 753,
      "train_loss": 0.0009382403837258203,
      "validation_loss": null
    },
    {
      "epoch": 754,
      "train_loss": 0.0011899262252781126,
      "validation_loss": null
    },
    {
      "epoch": 755,
      "train_loss": 0.0019173805356129175,
      "validation_loss": null
    },
    {
      "epoch": 756,
      "train_loss": 0.002956896205432713,
      "validation_loss": null
    },
    {
      "epoch": 757,
      "train_loss": 0.0010387969022202822,
      "validation_loss": null
    },
    {
      "epoch": 758,
      "train_loss": 0.0013347026769123557,
      "validation_loss": null
    },
    {
      "epoch": 759,
      "train_loss": 0.0011321504328710337,
      "validation_loss": null
    },
    {
      "epoch": 760,
      "train_loss": 0.0011839783077852593,
      "validation_loss": null
    },
    {
      "epoch": 761,
      "train_loss": 0.0007304246019985941,
      "validation_loss": null
    },
    {
      "epoch": 762,
      "train_loss": 0.0006161216006148607,
      "validation_loss": null
    },
    {
      "epoch": 763,
      "train_loss": 0.0009781976146364792,
      "validation_loss": null
    },
    {
      "epoch": 764,
      "train_loss": 0.0007347292881200297,
      "validation_loss": null
    },
    {
      "epoch": 765,
      "train_loss": 0.0006227253599920206,
      "validation_loss": null
    },
    {
      "epoch": 766,
      "train_loss": 0.0005160938114083061,
      "validation_loss": null
    },
    {
      "epoch": 767,
      "train_loss": 0.00043149721912211843,
      "validation_loss": null
    },
    {
      "epoch": 768,
      "train_loss": 0.0004510220605880022,
      "validation_loss": null
    },
    {
      "epoch": 769,
      "train_loss": 0.0004925578509250449,
      "validation_loss": null
    },
    {
      "epoch": 770,
      "train_loss": 0.00033234662098240934,
      "validation_loss": null
    },
    {
      "epoch": 771,
      "train_loss": 0.0003269736254070368,
      "validation_loss": null
    },
    {
      "epoch": 772,
      "train_loss": 0.00024978688452392817,
      "validation_loss": null
    },
    {
      "epoch": 773,
      "train_loss": 0.00023271632057407664,
      "validation_loss": null
    },
    {
      "epoch": 774,
      "train_loss": 0.000278796528517786,
      "validation_loss": null
    },
    {
      "epoch": 775,
      "train_loss": 0.00033003553916286264,
      "validation_loss": null
    },
    {
      "epoch": 776,
      "train_loss": 0.000304506158701972,
      "validation_loss": null
    },
    {
      "epoch": 777,
      "train_loss": 0.00030131866191772535,
      "validation_loss": null
    },
    {
      "epoch": 778,
      "train_loss": 0.00023803401563782245,
      "validation_loss": null
    },
    {
      "epoch": 779,
      "train_loss": 0.000280683704962333,
      "validation_loss": null
    },
    {
      "epoch": 780,
      "train_loss": 0.00030349250947539177,
      "validation_loss": null
    },
    {
      "epoch": 781,
      "train_loss": 0.00035689636165948794,
      "validation_loss": null
    },
    {
      "epoch": 782,
      "train_loss": 0.0003814356168732047,
      "validation_loss": null
    },
    {
      "epoch": 783,
      "train_loss": 0.00034775519775899336,
      "validation_loss": null
    },
    {
      "epoch": 784,
      "train_loss": 0.00022338080452755094,
      "validation_loss": null
    },
    {
      "epoch": 785,
      "train_loss": 0.0002386726232038604,
      "validation_loss": null
    },
    {
      "epoch": 786,
      "train_loss": 0.00019253717618994415,
      "validation_loss": null
    },
    {
      "epoch": 787,
      "train_loss": 0.00018436805920727138,
      "validation_loss": null
    },
    {
      "epoch": 788,
      "train_loss": 0.00021425253685770763,
      "validation_loss": null
    },
    {
      "epoch": 789,
      "train_loss": 0.00020730667650544396,
      "validation_loss": null
    },
    {
      "epoch": 790,
      "train_loss": 0.00020977348807112625,
      "validation_loss": null
    },
    {
      "epoch": 791,
      "train_loss": 0.0002652908442541957,
      "validation_loss": null
    },
    {
      "epoch": 792,
      "train_loss": 0.0005175701484808491,
      "validation_loss": null
    },
    {
      "epoch": 793,
      "train_loss": 0.0006401628909063422,
      "validation_loss": null
    },
    {
      "epoch": 794,
      "train_loss": 0.0007171220074976898,
      "validation_loss": null
    },
    {
      "epoch": 795,
      "train_loss": 0.0023950391332618892,
      "validation_loss": null
    },
    {
      "epoch": 796,
      "train_loss": 0.006782756898448699,
      "validation_loss": null
    },
    {
      "epoch": 797,
      "train_loss": 0.002877043786510411,
      "validation_loss": null
    },
    {
      "epoch": 798,
      "train_loss": 0.002455259834985352,
      "validation_loss": null
    },
    {
      "epoch": 799,
      "train_loss": 0.0015917569528230363,
      "validation_loss": null
    },
    {
      "epoch": 800,
      "train_loss": 0.0007557808680252896,
      "validation_loss": null
    },
    {
      "epoch": 801,
      "train_loss": 0.0005920583822040095,
      "validation_loss": null
    },
    {
      "epoch": 802,
      "train_loss": 0.0005891586349914885,
      "validation_loss": null
    },
    {
      "epoch": 803,
      "train_loss": 0.00031006305006384436,
      "validation_loss": null
    },
    {
      "epoch": 804,
      "train_loss": 0.00028543956513102684,
      "validation_loss": null
    },
    {
      "epoch": 805,
      "train_loss": 0.00031684713354075537,
      "validation_loss": null
    },
    {
      "epoch": 806,
      "train_loss": 0.0002987208895825057,
      "validation_loss": null
    },
    {
      "epoch": 807,
      "train_loss": 0.0002662780849883954,
      "validation_loss": null
    },
    {
      "epoch": 808,
      "train_loss": 0.0002354408206883818,
      "validation_loss": null
    },
    {
      "epoch": 809,
      "train_loss": 0.00023052485873146603,
      "validation_loss": null
    },
    {
      "epoch": 810,
      "train_loss": 0.0002278791022642205,
      "validation_loss": null
    },
    {
      "epoch": 811,
      "train_loss": 0.00019913560148173323,
      "validation_loss": null
    },
    {
      "epoch": 812,
      "train_loss": 0.00025064574664510373,
      "validation_loss": null
    },
    {
      "epoch": 813,
      "train_loss": 0.0002185543918231916,
      "validation_loss": null
    },
    {
      "epoch": 814,
      "train_loss": 0.00021072885849409632,
      "validation_loss": null
    },
    {
      "epoch": 815,
      "train_loss": 0.00017122174580840187,
      "validation_loss": null
    },
    {
      "epoch": 816,
      "train_loss": 0.00016566852314604653,
      "validation_loss": null
    },
    {
      "epoch": 817,
      "train_loss": 0.00016370512215265384,
      "validation_loss": null
    },
    {
      "epoch": 818,
      "train_loss": 0.00015906461501597532,
      "validation_loss": null
    },
    {
      "epoch": 819,
      "train_loss": 0.00016297961459107077,
      "validation_loss": null
    },
    {
      "epoch": 820,
      "train_loss": 0.0001482236144107042,
      "validation_loss": null
    },
    {
      "epoch": 821,
      "train_loss": 0.00016527646382908442,
      "validation_loss": null
    },
    {
      "epoch": 822,
      "train_loss": 0.00018056745789686425,
      "validation_loss": null
    },
    {
      "epoch": 823,
      "train_loss": 0.00019324578978638683,
      "validation_loss": null
    },
    {
      "epoch": 824,
      "train_loss": 0.00017295545452119163,
      "validation_loss": null
    },
    {
      "epoch": 825,
      "train_loss": 0.00014172451159942689,
      "validation_loss": null
    },
    {
      "epoch": 826,
      "train_loss": 0.00016734484476425374,
      "validation_loss": null
    },
    {
      "epoch": 827,
      "train_loss": 0.0001731188127046658,
      "validation_loss": null
    },
    {
      "epoch": 828,
      "train_loss": 0.00022606601366229975,
      "validation_loss": null
    },
    {
      "epoch": 829,
      "train_loss": 0.00021842342620301578,
      "validation_loss": null
    },
    {
      "epoch": 830,
      "train_loss": 0.0002718572957544691,
      "validation_loss": null
    },
    {
      "epoch": 831,
      "train_loss": 0.00035180481305966776,
      "validation_loss": null
    },
    {
      "epoch": 832,
      "train_loss": 0.00035604319418780506,
      "validation_loss": null
    },
    {
      "epoch": 833,
      "train_loss": 0.0002399059869478353,
      "validation_loss": null
    },
    {
      "epoch": 834,
      "train_loss": 0.0001654912129096273,
      "validation_loss": null
    },
    {
      "epoch": 835,
      "train_loss": 0.00017290432555859702,
      "validation_loss": null
    },
    {
      "epoch": 836,
      "train_loss": 0.0001694295044419252,
      "validation_loss": null
    },
    {
      "epoch": 837,
      "train_loss": 0.00015974878462859325,
      "validation_loss": null
    },
    {
      "epoch": 838,
      "train_loss": 0.0001741629077716627,
      "validation_loss": null
    },
    {
      "epoch": 839,
      "train_loss": 0.0002136170029795418,
      "validation_loss": null
    },
    {
      "epoch": 840,
      "train_loss": 0.00017054567352816876,
      "validation_loss": null
    },
    {
      "epoch": 841,
      "train_loss": 0.00016683604917488992,
      "validation_loss": null
    },
    {
      "epoch": 842,
      "train_loss": 0.0001308344572963607,
      "validation_loss": null
    },
    {
      "epoch": 843,
      "train_loss": 0.00014800823474716808,
      "validation_loss": null
    },
    {
      "epoch": 844,
      "train_loss": 0.00016199477411444404,
      "validation_loss": null
    },
    {
      "epoch": 845,
      "train_loss": 0.00014213782398858003,
      "validation_loss": null
    },
    {
      "epoch": 846,
      "train_loss": 0.00012883218652051356,
      "validation_loss": null
    },
    {
      "epoch": 847,
      "train_loss": 0.00014553948858520016,
      "validation_loss": null
    },
    {
      "epoch": 848,
      "train_loss": 0.00012460226813952127,
      "validation_loss": null
    },
    {
      "epoch": 849,
      "train_loss": 0.00011836223873413271,
      "validation_loss": null
    },
    {
      "epoch": 850,
      "train_loss": 0.00012314305705432262,
      "validation_loss": null
    },
    {
      "epoch": 851,
      "train_loss": 0.00016483756593920084,
      "validation_loss": null
    },
    {
      "epoch": 852,
      "train_loss": 0.0002152589828862498,
      "validation_loss": null
    },
    {
      "epoch": 853,
      "train_loss": 0.00027572549394487095,
      "validation_loss": null
    },
    {
      "epoch": 854,
      "train_loss": 0.0002248421256404577,
      "validation_loss": null
    },
    {
      "epoch": 855,
      "train_loss": 0.00026306251270903484,
      "validation_loss": null
    },
    {
      "epoch": 856,
      "train_loss": 0.00038034455630824796,
      "validation_loss": null
    },
    {
      "epoch": 857,
      "train_loss": 0.0003628056211810973,
      "validation_loss": null
    },
    {
      "epoch": 858,
      "train_loss": 0.00042834066294340626,
      "validation_loss": null
    },
    {
      "epoch": 859,
      "train_loss": 0.0005555782441256775,
      "validation_loss": null
    },
    {
      "epoch": 860,
      "train_loss": 0.0005816754419356585,
      "validation_loss": null
    },
    {
      "epoch": 861,
      "train_loss": 0.0003518485076104601,
      "validation_loss": null
    },
    {
      "epoch": 862,
      "train_loss": 0.000320797280033326,
      "validation_loss": null
    },
    {
      "epoch": 863,
      "train_loss": 0.0002325587621372607,
      "validation_loss": null
    },
    {
      "epoch": 864,
      "train_loss": 0.00023307070174875358,
      "validation_loss": null
    },
    {
      "epoch": 865,
      "train_loss": 0.0003831831272691488,
      "validation_loss": null
    },
    {
      "epoch": 866,
      "train_loss": 0.0011998176907784203,
      "validation_loss": null
    },
    {
      "epoch": 867,
      "train_loss": 0.002335145539190206,
      "validation_loss": null
    },
    {
      "epoch": 868,
      "train_loss": 0.003093706317789232,
      "validation_loss": null
    },
    {
      "epoch": 869,
      "train_loss": 0.003105526117400991,
      "validation_loss": null
    },
    {
      "epoch": 870,
      "train_loss": 0.002433285750966105,
      "validation_loss": null
    },
    {
      "epoch": 871,
      "train_loss": 0.004401622029642264,
      "validation_loss": null
    },
    {
      "epoch": 872,
      "train_loss": 0.004604538669809699,
      "validation_loss": null
    },
    {
      "epoch": 873,
      "train_loss": 0.004753688751305971,
      "validation_loss": null
    },
    {
      "epoch": 874,
      "train_loss": 0.0020904788721559774,
      "validation_loss": null
    },
    {
      "epoch": 875,
      "train_loss": 0.0021345108333561155,
      "validation_loss": null
    },
    {
      "epoch": 876,
      "train_loss": 0.0015234127576049003,
      "validation_loss": null
    },
    {
      "epoch": 877,
      "train_loss": 0.0011133849297443198,
      "validation_loss": null
    },
    {
      "epoch": 878,
      "train_loss": 0.0023919305401957696,
      "validation_loss": null
    },
    {
      "epoch": 879,
      "train_loss": 0.002571308985352516,
      "validation_loss": null
    },
    {
      "epoch": 880,
      "train_loss": 0.0035548838528080117,
      "validation_loss": null
    },
    {
      "epoch": 881,
      "train_loss": 0.0016825429370833768,
      "validation_loss": null
    },
    {
      "epoch": 882,
      "train_loss": 0.002314870122871879,
      "validation_loss": null
    },
    {
      "epoch": 883,
      "train_loss": 0.0013613532387858464,
      "validation_loss": null
    },
    {
      "epoch": 884,
      "train_loss": 0.000703321223328304,
      "validation_loss": null
    },
    {
      "epoch": 885,
      "train_loss": 0.000907327193999663,
      "validation_loss": null
    },
    {
      "epoch": 886,
      "train_loss": 0.0007581761682457808,
      "validation_loss": null
    },
    {
      "epoch": 887,
      "train_loss": 0.00044360799236326583,
      "validation_loss": null
    },
    {
      "epoch": 888,
      "train_loss": 0.0004174150154641312,
      "validation_loss": null
    },
    {
      "epoch": 889,
      "train_loss": 0.0002610015168708439,
      "validation_loss": null
    },
    {
      "epoch": 890,
      "train_loss": 0.00026355588731045526,
      "validation_loss": null
    },
    {
      "epoch": 891,
      "train_loss": 0.00033264199414083525,
      "validation_loss": null
    },
    {
      "epoch": 892,
      "train_loss": 0.00027097188285551965,
      "validation_loss": null
    },
    {
      "epoch": 893,
      "train_loss": 0.00017509884815404398,
      "validation_loss": null
    },
    {
      "epoch": 894,
      "train_loss": 0.0001770055856064169,
      "validation_loss": null
    },
    {
      "epoch": 895,
      "train_loss": 0.00017086080495371588,
      "validation_loss": null
    },
    {
      "epoch": 896,
      "train_loss": 0.00015337311277916242,
      "validation_loss": null
    },
    {
      "epoch": 897,
      "train_loss": 0.0001462206960746294,
      "validation_loss": null
    },
    {
      "epoch": 898,
      "train_loss": 0.0001518058852525428,
      "validation_loss": null
    },
    {
      "epoch": 899,
      "train_loss": 0.00013676786814661077,
      "validation_loss": null
    },
    {
      "epoch": 900,
      "train_loss": 0.00013986807445892028,
      "validation_loss": null
    },
    {
      "epoch": 901,
      "train_loss": 0.00013203190287135335,
      "validation_loss": null
    },
    {
      "epoch": 902,
      "train_loss": 0.00014850352292544104,
      "validation_loss": null
    },
    {
      "epoch": 903,
      "train_loss": 0.00011670999811030924,
      "validation_loss": null
    },
    {
      "epoch": 904,
      "train_loss": 0.00011150503754227732,
      "validation_loss": null
    },
    {
      "epoch": 905,
      "train_loss": 0.00011787676824800049,
      "validation_loss": null
    },
    {
      "epoch": 906,
      "train_loss": 0.00012276749799235,
      "validation_loss": null
    },
    {
      "epoch": 907,
      "train_loss": 0.00012374504876788706,
      "validation_loss": null
    },
    {
      "epoch": 908,
      "train_loss": 0.00011049024033127353,
      "validation_loss": null
    },
    {
      "epoch": 909,
      "train_loss": 0.00012776110492672565,
      "validation_loss": null
    },
    {
      "epoch": 910,
      "train_loss": 0.00011733310788662898,
      "validation_loss": null
    },
    {
      "epoch": 911,
      "train_loss": 0.0001365665124871561,
      "validation_loss": null
    },
    {
      "epoch": 912,
      "train_loss": 0.00015429806323178732,
      "validation_loss": null
    },
    {
      "epoch": 913,
      "train_loss": 0.00014843365675511045,
      "validation_loss": null
    },
    {
      "epoch": 914,
      "train_loss": 0.00015163383876723755,
      "validation_loss": null
    },
    {
      "epoch": 915,
      "train_loss": 0.00018156627629650757,
      "validation_loss": null
    },
    {
      "epoch": 916,
      "train_loss": 0.00020324837694513716,
      "validation_loss": null
    },
    {
      "epoch": 917,
      "train_loss": 0.0002101120246354387,
      "validation_loss": null
    },
    {
      "epoch": 918,
      "train_loss": 0.00028824223328330036,
      "validation_loss": null
    },
    {
      "epoch": 919,
      "train_loss": 0.0002291210985276848,
      "validation_loss": null
    },
    {
      "epoch": 920,
      "train_loss": 0.00030586248289586767,
      "validation_loss": null
    },
    {
      "epoch": 921,
      "train_loss": 0.0002423776847232754,
      "validation_loss": null
    },
    {
      "epoch": 922,
      "train_loss": 0.00023081547503049174,
      "validation_loss": null
    },
    {
      "epoch": 923,
      "train_loss": 0.00023825817172312073,
      "validation_loss": null
    },
    {
      "epoch": 924,
      "train_loss": 0.0002040683054171192,
      "validation_loss": null
    },
    {
      "epoch": 925,
      "train_loss": 0.00019640317542426701,
      "validation_loss": null
    },
    {
      "epoch": 926,
      "train_loss": 0.0002084882912135476,
      "validation_loss": null
    },
    {
      "epoch": 927,
      "train_loss": 0.00014425702102016658,
      "validation_loss": null
    },
    {
      "epoch": 928,
      "train_loss": 0.00010090974683407694,
      "validation_loss": null
    },
    {
      "epoch": 929,
      "train_loss": 9.94203445669781e-05,
      "validation_loss": null
    },
    {
      "epoch": 930,
      "train_loss": 0.00010322426371405729,
      "validation_loss": null
    },
    {
      "epoch": 931,
      "train_loss": 0.00010219262023585745,
      "validation_loss": null
    },
    {
      "epoch": 932,
      "train_loss": 9.103539115231897e-05,
      "validation_loss": null
    },
    {
      "epoch": 933,
      "train_loss": 0.00010297558522627999,
      "validation_loss": null
    },
    {
      "epoch": 934,
      "train_loss": 9.484860412259068e-05,
      "validation_loss": null
    },
    {
      "epoch": 935,
      "train_loss": 9.47902501340852e-05,
      "validation_loss": null
    },
    {
      "epoch": 936,
      "train_loss": 9.444029516695689e-05,
      "validation_loss": null
    },
    {
      "epoch": 937,
      "train_loss": 9.037670113482616e-05,
      "validation_loss": null
    },
    {
      "epoch": 938,
      "train_loss": 9.457454264823657e-05,
      "validation_loss": null
    },
    {
      "epoch": 939,
      "train_loss": 0.00010840673348866403,
      "validation_loss": null
    },
    {
      "epoch": 940,
      "train_loss": 0.00012280081298538588,
      "validation_loss": null
    },
    {
      "epoch": 941,
      "train_loss": 0.000150383886648342,
      "validation_loss": null
    },
    {
      "epoch": 942,
      "train_loss": 0.0001551491740327846,
      "validation_loss": null
    },
    {
      "epoch": 943,
      "train_loss": 0.00011972349764417029,
      "validation_loss": null
    },
    {
      "epoch": 944,
      "train_loss": 0.00010174685222510662,
      "validation_loss": null
    },
    {
      "epoch": 945,
      "train_loss": 9.132473335032248e-05,
      "validation_loss": null
    },
    {
      "epoch": 946,
      "train_loss": 8.111527171826715e-05,
      "validation_loss": null
    },
    {
      "epoch": 947,
      "train_loss": 8.04136191921619e-05,
      "validation_loss": null
    },
    {
      "epoch": 948,
      "train_loss": 8.244429207277588e-05,
      "validation_loss": null
    },
    {
      "epoch": 949,
      "train_loss": 8.430655563845196e-05,
      "validation_loss": null
    },
    {
      "epoch": 950,
      "train_loss": 9.969355515851121e-05,
      "validation_loss": null
    },
    {
      "epoch": 951,
      "train_loss": 9.748316450794745e-05,
      "validation_loss": null
    },
    {
      "epoch": 952,
      "train_loss": 9.260390086435816e-05,
      "validation_loss": null
    },
    {
      "epoch": 953,
      "train_loss": 9.253850374888215e-05,
      "validation_loss": null
    },
    {
      "epoch": 954,
      "train_loss": 9.707082143803645e-05,
      "validation_loss": null
    },
    {
      "epoch": 955,
      "train_loss": 0.0001035311741159401,
      "validation_loss": null
    },
    {
      "epoch": 956,
      "train_loss": 8.841534711084225e-05,
      "validation_loss": null
    },
    {
      "epoch": 957,
      "train_loss": 0.00010434578467134593,
      "validation_loss": null
    },
    {
      "epoch": 958,
      "train_loss": 9.130590802265538e-05,
      "validation_loss": null
    },
    {
      "epoch": 959,
      "train_loss": 0.00010281490702052704,
      "validation_loss": null
    },
    {
      "epoch": 960,
      "train_loss": 0.00010776497624141889,
      "validation_loss": null
    },
    {
      "epoch": 961,
      "train_loss": 0.00010270731096776824,
      "validation_loss": null
    },
    {
      "epoch": 962,
      "train_loss": 0.00010613106941390369,
      "validation_loss": null
    },
    {
      "epoch": 963,
      "train_loss": 0.00010140345693798736,
      "validation_loss": null
    },
    {
      "epoch": 964,
      "train_loss": 8.80876856778438e-05,
      "validation_loss": null
    },
    {
      "epoch": 965,
      "train_loss": 9.329639983156489e-05,
      "validation_loss": null
    },
    {
      "epoch": 966,
      "train_loss": 0.0001175190069867919,
      "validation_loss": null
    },
    {
      "epoch": 967,
      "train_loss": 9.199162094672729e-05,
      "validation_loss": null
    },
    {
      "epoch": 968,
      "train_loss": 0.00010304630632163025,
      "validation_loss": null
    },
    {
      "epoch": 969,
      "train_loss": 9.438858453197301e-05,
      "validation_loss": null
    },
    {
      "epoch": 970,
      "train_loss": 9.99228023930401e-05,
      "validation_loss": null
    },
    {
      "epoch": 971,
      "train_loss": 0.00017065850746197006,
      "validation_loss": null
    },
    {
      "epoch": 972,
      "train_loss": 0.00017861610588928065,
      "validation_loss": null
    },
    {
      "epoch": 973,
      "train_loss": 0.00013187438889872283,
      "validation_loss": null
    },
    {
      "epoch": 974,
      "train_loss": 0.00015500454658952853,
      "validation_loss": null
    },
    {
      "epoch": 975,
      "train_loss": 0.00010349154116637591,
      "validation_loss": null
    },
    {
      "epoch": 976,
      "train_loss": 0.000123929325329502,
      "validation_loss": null
    },
    {
      "epoch": 977,
      "train_loss": 0.00010580920206848532,
      "validation_loss": null
    },
    {
      "epoch": 978,
      "train_loss": 0.00014269577352226607,
      "validation_loss": null
    },
    {
      "epoch": 979,
      "train_loss": 0.00019094496868395558,
      "validation_loss": null
    },
    {
      "epoch": 980,
      "train_loss": 0.00014749633313234275,
      "validation_loss": null
    },
    {
      "epoch": 981,
      "train_loss": 0.00014802241558855813,
      "validation_loss": null
    },
    {
      "epoch": 982,
      "train_loss": 0.00036500928399618715,
      "validation_loss": null
    },
    {
      "epoch": 983,
      "train_loss": 0.0005307779364456008,
      "validation_loss": null
    },
    {
      "epoch": 984,
      "train_loss": 0.0005097830241235594,
      "validation_loss": null
    },
    {
      "epoch": 985,
      "train_loss": 0.000380069301981065,
      "validation_loss": null
    },
    {
      "epoch": 986,
      "train_loss": 0.00025922583396701765,
      "validation_loss": null
    },
    {
      "epoch": 987,
      "train_loss": 0.0002623970495834429,
      "validation_loss": null
    },
    {
      "epoch": 988,
      "train_loss": 0.00033397374661742814,
      "validation_loss": null
    },
    {
      "epoch": 989,
      "train_loss": 0.00030787229964820045,
      "validation_loss": null
    },
    {
      "epoch": 990,
      "train_loss": 0.000446940106081052,
      "validation_loss": null
    },
    {
      "epoch": 991,
      "train_loss": 0.0002733182401344594,
      "validation_loss": null
    },
    {
      "epoch": 992,
      "train_loss": 0.0002458218110000922,
      "validation_loss": null
    },
    {
      "epoch": 993,
      "train_loss": 0.0002747256011288199,
      "validation_loss": null
    },
    {
      "epoch": 994,
      "train_loss": 0.0003239854599491486,
      "validation_loss": null
    },
    {
      "epoch": 995,
      "train_loss": 0.0003353563790571772,
      "validation_loss": null
    },
    {
      "epoch": 996,
      "train_loss": 0.000724314886610955,
      "validation_loss": null
    },
    {
      "epoch": 997,
      "train_loss": 0.0014527407668841381,
      "validation_loss": null
    },
    {
      "epoch": 998,
      "train_loss": 0.0008559623804305576,
      "validation_loss": null
    },
    {
      "epoch": 999,
      "train_loss": 0.0011583124481452007,
      "validation_loss": null
    },
    {
      "epoch": 1000,
      "train_loss": 0.0013756217894196096,
      "validation_loss": null
    },
    {
      "epoch": 1001,
      "train_loss": 0.0006880188949354407,
      "validation_loss": null
    },
    {
      "epoch": 1002,
      "train_loss": 0.0004679220631563415,
      "validation_loss": null
    },
    {
      "epoch": 1003,
      "train_loss": 0.0004885263818626603,
      "validation_loss": null
    },
    {
      "epoch": 1004,
      "train_loss": 0.00045133101715085405,
      "validation_loss": null
    },
    {
      "epoch": 1005,
      "train_loss": 0.00035144941163404536,
      "validation_loss": null
    },
    {
      "epoch": 1006,
      "train_loss": 0.00022349132147307196,
      "validation_loss": null
    },
    {
      "epoch": 1007,
      "train_loss": 0.00016289198699976422,
      "validation_loss": null
    },
    {
      "epoch": 1008,
      "train_loss": 0.00014415876487166516,
      "validation_loss": null
    },
    {
      "epoch": 1009,
      "train_loss": 0.00012512632787951993,
      "validation_loss": null
    },
    {
      "epoch": 1010,
      "train_loss": 0.00012711184293342134,
      "validation_loss": null
    },
    {
      "epoch": 1011,
      "train_loss": 0.00011445916687888611,
      "validation_loss": null
    },
    {
      "epoch": 1012,
      "train_loss": 0.00012197510901993762,
      "validation_loss": null
    },
    {
      "epoch": 1013,
      "train_loss": 0.0001648931033236699,
      "validation_loss": null
    },
    {
      "epoch": 1014,
      "train_loss": 0.0001986397740741571,
      "validation_loss": null
    },
    {
      "epoch": 1015,
      "train_loss": 0.0003557902705829798,
      "validation_loss": null
    },
    {
      "epoch": 1016,
      "train_loss": 0.00037111025044901506,
      "validation_loss": null
    },
    {
      "epoch": 1017,
      "train_loss": 0.00018635455813233016,
      "validation_loss": null
    },
    {
      "epoch": 1018,
      "train_loss": 0.0004877220069627381,
      "validation_loss": null
    },
    {
      "epoch": 1019,
      "train_loss": 0.0004469698243257072,
      "validation_loss": null
    },
    {
      "epoch": 1020,
      "train_loss": 0.0009439836042777946,
      "validation_loss": null
    },
    {
      "epoch": 1021,
      "train_loss": 0.0011115643169937863,
      "validation_loss": null
    },
    {
      "epoch": 1022,
      "train_loss": 0.000938103421099691,
      "validation_loss": null
    },
    {
      "epoch": 1023,
      "train_loss": 0.0004350223171059042,
      "validation_loss": null
    },
    {
      "epoch": 1024,
      "train_loss": 0.00031734669108926837,
      "validation_loss": null
    },
    {
      "epoch": 1025,
      "train_loss": 0.0002760576737475478,
      "validation_loss": null
    },
    {
      "epoch": 1026,
      "train_loss": 0.00017973668420583836,
      "validation_loss": null
    },
    {
      "epoch": 1027,
      "train_loss": 0.0002334660562660752,
      "validation_loss": null
    },
    {
      "epoch": 1028,
      "train_loss": 0.00041094676998909563,
      "validation_loss": null
    },
    {
      "epoch": 1029,
      "train_loss": 0.00044012074229410954,
      "validation_loss": null
    },
    {
      "epoch": 1030,
      "train_loss": 0.00026967984093870554,
      "validation_loss": null
    },
    {
      "epoch": 1031,
      "train_loss": 0.00026575331543830945,
      "validation_loss": null
    },
    {
      "epoch": 1032,
      "train_loss": 0.0002230225331408696,
      "validation_loss": null
    },
    {
      "epoch": 1033,
      "train_loss": 0.0001940097524008403,
      "validation_loss": null
    },
    {
      "epoch": 1034,
      "train_loss": 0.00012612865409917303,
      "validation_loss": null
    },
    {
      "epoch": 1035,
      "train_loss": 9.661035356758576e-05,
      "validation_loss": null
    },
    {
      "epoch": 1036,
      "train_loss": 7.517793185090543e-05,
      "validation_loss": null
    },
    {
      "epoch": 1037,
      "train_loss": 9.127309183693594e-05,
      "validation_loss": null
    },
    {
      "epoch": 1038,
      "train_loss": 0.00024587628690318926,
      "validation_loss": null
    },
    {
      "epoch": 1039,
      "train_loss": 0.0002232526061965877,
      "validation_loss": null
    },
    {
      "epoch": 1040,
      "train_loss": 0.00018456172175420862,
      "validation_loss": null
    },
    {
      "epoch": 1041,
      "train_loss": 0.00018226966818070246,
      "validation_loss": null
    },
    {
      "epoch": 1042,
      "train_loss": 0.00019489023139208762,
      "validation_loss": null
    },
    {
      "epoch": 1043,
      "train_loss": 8.725250194806399e-05,
      "validation_loss": null
    },
    {
      "epoch": 1044,
      "train_loss": 9.718785440782085e-05,
      "validation_loss": null
    },
    {
      "epoch": 1045,
      "train_loss": 0.00013351299882439585,
      "validation_loss": null
    },
    {
      "epoch": 1046,
      "train_loss": 0.00013049348086092828,
      "validation_loss": null
    },
    {
      "epoch": 1047,
      "train_loss": 0.0002065470131735007,
      "validation_loss": null
    },
    {
      "epoch": 1048,
      "train_loss": 0.00029490028908993636,
      "validation_loss": null
    },
    {
      "epoch": 1049,
      "train_loss": 0.00030565403479461867,
      "validation_loss": null
    },
    {
      "epoch": 1050,
      "train_loss": 0.00015936946308809437,
      "validation_loss": null
    },
    {
      "epoch": 1051,
      "train_loss": 0.00013137520323248787,
      "validation_loss": null
    },
    {
      "epoch": 1052,
      "train_loss": 0.00011915031953119776,
      "validation_loss": null
    },
    {
      "epoch": 1053,
      "train_loss": 0.00011966653174668964,
      "validation_loss": null
    },
    {
      "epoch": 1054,
      "train_loss": 0.00010951341876837735,
      "validation_loss": null
    },
    {
      "epoch": 1055,
      "train_loss": 7.808235709995238e-05,
      "validation_loss": null
    },
    {
      "epoch": 1056,
      "train_loss": 0.00013473586942483153,
      "validation_loss": null
    },
    {
      "epoch": 1057,
      "train_loss": 0.00018082471150490973,
      "validation_loss": null
    },
    {
      "epoch": 1058,
      "train_loss": 0.00011048302662351893,
      "validation_loss": null
    },
    {
      "epoch": 1059,
      "train_loss": 0.00010141401758624447,
      "validation_loss": null
    },
    {
      "epoch": 1060,
      "train_loss": 0.0001012602725596581,
      "validation_loss": null
    },
    {
      "epoch": 1061,
      "train_loss": 0.00010160534535922731,
      "validation_loss": null
    },
    {
      "epoch": 1062,
      "train_loss": 0.00021909308917303052,
      "validation_loss": null
    },
    {
      "epoch": 1063,
      "train_loss": 0.0003295760058487455,
      "validation_loss": null
    },
    {
      "epoch": 1064,
      "train_loss": 0.000239842627908931,
      "validation_loss": null
    },
    {
      "epoch": 1065,
      "train_loss": 0.00021672758157365024,
      "validation_loss": null
    },
    {
      "epoch": 1066,
      "train_loss": 0.00011403203340402494,
      "validation_loss": null
    },
    {
      "epoch": 1067,
      "train_loss": 0.00011924905750978117,
      "validation_loss": null
    },
    {
      "epoch": 1068,
      "train_loss": 9.240382577344362e-05,
      "validation_loss": null
    },
    {
      "epoch": 1069,
      "train_loss": 8.889534122621019e-05,
      "validation_loss": null
    },
    {
      "epoch": 1070,
      "train_loss": 8.246529331598949e-05,
      "validation_loss": null
    },
    {
      "epoch": 1071,
      "train_loss": 8.322256892117568e-05,
      "validation_loss": null
    },
    {
      "epoch": 1072,
      "train_loss": 0.00013319443718905354,
      "validation_loss": null
    },
    {
      "epoch": 1073,
      "train_loss": 0.00012227961148407,
      "validation_loss": null
    },
    {
      "epoch": 1074,
      "train_loss": 0.00010457126740624921,
      "validation_loss": null
    },
    {
      "epoch": 1075,
      "train_loss": 9.744077275778788e-05,
      "validation_loss": null
    },
    {
      "epoch": 1076,
      "train_loss": 0.00015328154646946737,
      "validation_loss": null
    },
    {
      "epoch": 1077,
      "train_loss": 0.00022374575928552076,
      "validation_loss": null
    },
    {
      "epoch": 1078,
      "train_loss": 0.00014817376439977024,
      "validation_loss": null
    },
    {
      "epoch": 1079,
      "train_loss": 0.0001630213535261444,
      "validation_loss": null
    },
    {
      "epoch": 1080,
      "train_loss": 0.00014917503883932822,
      "validation_loss": null
    },
    {
      "epoch": 1081,
      "train_loss": 0.00015114185872435983,
      "validation_loss": null
    },
    {
      "epoch": 1082,
      "train_loss": 0.0001463778228046269,
      "validation_loss": null
    },
    {
      "epoch": 1083,
      "train_loss": 0.0001446905573377282,
      "validation_loss": null
    },
    {
      "epoch": 1084,
      "train_loss": 0.0001839279219792742,
      "validation_loss": null
    },
    {
      "epoch": 1085,
      "train_loss": 0.0002627584237618268,
      "validation_loss": null
    },
    {
      "epoch": 1086,
      "train_loss": 0.0004898756322088755,
      "validation_loss": null
    },
    {
      "epoch": 1087,
      "train_loss": 0.0005575583879059801,
      "validation_loss": null
    },
    {
      "epoch": 1088,
      "train_loss": 0.001444319644684179,
      "validation_loss": null
    },
    {
      "epoch": 1089,
      "train_loss": 0.0012509823395198004,
      "validation_loss": null
    },
    {
      "epoch": 1090,
      "train_loss": 0.0009449137529979149,
      "validation_loss": null
    },
    {
      "epoch": 1091,
      "train_loss": 0.0009749911065834264,
      "validation_loss": null
    },
    {
      "epoch": 1092,
      "train_loss": 0.0010113204771187156,
      "validation_loss": null
    },
    {
      "epoch": 1093,
      "train_loss": 0.001235517375688586,
      "validation_loss": null
    },
    {
      "epoch": 1094,
      "train_loss": 0.0008492101753492736,
      "validation_loss": null
    },
    {
      "epoch": 1095,
      "train_loss": 0.0004617009916627366,
      "validation_loss": null
    },
    {
      "epoch": 1096,
      "train_loss": 0.00021343522530514747,
      "validation_loss": null
    },
    {
      "epoch": 1097,
      "train_loss": 0.00027656806519694836,
      "validation_loss": null
    },
    {
      "epoch": 1098,
      "train_loss": 0.0006183139436567823,
      "validation_loss": null
    },
    {
      "epoch": 1099,
      "train_loss": 0.001266878977831867,
      "validation_loss": null
    },
    {
      "epoch": 1100,
      "train_loss": 0.0009298383730411944,
      "validation_loss": null
    },
    {
      "epoch": 1101,
      "train_loss": 0.0007904918909642018,
      "validation_loss": null
    },
    {
      "epoch": 1102,
      "train_loss": 0.0003354157007480454,
      "validation_loss": null
    },
    {
      "epoch": 1103,
      "train_loss": 0.00027961267915088683,
      "validation_loss": null
    },
    {
      "epoch": 1104,
      "train_loss": 0.00023959041573107243,
      "validation_loss": null
    },
    {
      "epoch": 1105,
      "train_loss": 0.00023041137642899735,
      "validation_loss": null
    },
    {
      "epoch": 1106,
      "train_loss": 0.0002529116771054558,
      "validation_loss": null
    },
    {
      "epoch": 1107,
      "train_loss": 0.0002111959418268978,
      "validation_loss": null
    },
    {
      "epoch": 1108,
      "train_loss": 0.0001881355419754982,
      "validation_loss": null
    },
    {
      "epoch": 1109,
      "train_loss": 0.00014803741861315857,
      "validation_loss": null
    },
    {
      "epoch": 1110,
      "train_loss": 0.000120466462653389,
      "validation_loss": null
    },
    {
      "epoch": 1111,
      "train_loss": 9.560625605647349e-05,
      "validation_loss": null
    },
    {
      "epoch": 1112,
      "train_loss": 0.00010731548779100801,
      "validation_loss": null
    },
    {
      "epoch": 1113,
      "train_loss": 0.00011131537919734708,
      "validation_loss": null
    },
    {
      "epoch": 1114,
      "train_loss": 7.422730494807992e-05,
      "validation_loss": null
    },
    {
      "epoch": 1115,
      "train_loss": 6.999182005529292e-05,
      "validation_loss": null
    },
    {
      "epoch": 1116,
      "train_loss": 8.063937356281612e-05,
      "validation_loss": null
    },
    {
      "epoch": 1117,
      "train_loss": 8.257204414500545e-05,
      "validation_loss": null
    },
    {
      "epoch": 1118,
      "train_loss": 5.639478856917574e-05,
      "validation_loss": null
    },
    {
      "epoch": 1119,
      "train_loss": 5.461374934788587e-05,
      "validation_loss": null
    },
    {
      "epoch": 1120,
      "train_loss": 4.712108218680239e-05,
      "validation_loss": null
    },
    {
      "epoch": 1121,
      "train_loss": 6.18528659591296e-05,
      "validation_loss": null
    },
    {
      "epoch": 1122,
      "train_loss": 7.50206887055861e-05,
      "validation_loss": null
    },
    {
      "epoch": 1123,
      "train_loss": 7.926893118160983e-05,
      "validation_loss": null
    },
    {
      "epoch": 1124,
      "train_loss": 7.214688796213725e-05,
      "validation_loss": null
    },
    {
      "epoch": 1125,
      "train_loss": 5.8226850847454946e-05,
      "validation_loss": null
    },
    {
      "epoch": 1126,
      "train_loss": 6.126286785325242e-05,
      "validation_loss": null
    },
    {
      "epoch": 1127,
      "train_loss": 5.8463434672578136e-05,
      "validation_loss": null
    },
    {
      "epoch": 1128,
      "train_loss": 6.097617289116089e-05,
      "validation_loss": null
    },
    {
      "epoch": 1129,
      "train_loss": 6.76896298601706e-05,
      "validation_loss": null
    },
    {
      "epoch": 1130,
      "train_loss": 9.449734473795009e-05,
      "validation_loss": null
    },
    {
      "epoch": 1131,
      "train_loss": 0.00011582788768767689,
      "validation_loss": null
    },
    {
      "epoch": 1132,
      "train_loss": 7.31716018991493e-05,
      "validation_loss": null
    },
    {
      "epoch": 1133,
      "train_loss": 5.958925289127769e-05,
      "validation_loss": null
    },
    {
      "epoch": 1134,
      "train_loss": 6.92581840363952e-05,
      "validation_loss": null
    },
    {
      "epoch": 1135,
      "train_loss": 5.2197521856416846e-05,
      "validation_loss": null
    },
    {
      "epoch": 1136,
      "train_loss": 4.283811843682391e-05,
      "validation_loss": null
    },
    {
      "epoch": 1137,
      "train_loss": 4.604468793129652e-05,
      "validation_loss": null
    },
    {
      "epoch": 1138,
      "train_loss": 0.00010262739629575258,
      "validation_loss": null
    },
    {
      "epoch": 1139,
      "train_loss": 5.293549596293209e-05,
      "validation_loss": null
    },
    {
      "epoch": 1140,
      "train_loss": 5.828213276496778e-05,
      "validation_loss": null
    },
    {
      "epoch": 1141,
      "train_loss": 0.0001070873634792709,
      "validation_loss": null
    },
    {
      "epoch": 1142,
      "train_loss": 0.0001611163509854426,
      "validation_loss": null
    },
    {
      "epoch": 1143,
      "train_loss": 0.00018107403431915573,
      "validation_loss": null
    },
    {
      "epoch": 1144,
      "train_loss": 0.00021606507652904838,
      "validation_loss": null
    },
    {
      "epoch": 1145,
      "train_loss": 0.00012741058971086104,
      "validation_loss": null
    },
    {
      "epoch": 1146,
      "train_loss": 0.00013412601967704378,
      "validation_loss": null
    },
    {
      "epoch": 1147,
      "train_loss": 0.00010106392922655989,
      "validation_loss": null
    },
    {
      "epoch": 1148,
      "train_loss": 0.000129946660308633,
      "validation_loss": null
    },
    {
      "epoch": 1149,
      "train_loss": 0.00018322753587401367,
      "validation_loss": null
    },
    {
      "epoch": 1150,
      "train_loss": 0.00011032122670763379,
      "validation_loss": null
    },
    {
      "epoch": 1151,
      "train_loss": 0.0002860246822820045,
      "validation_loss": null
    },
    {
      "epoch": 1152,
      "train_loss": 0.0011877785105348772,
      "validation_loss": null
    },
    {
      "epoch": 1153,
      "train_loss": 0.0014504324224415338,
      "validation_loss": null
    },
    {
      "epoch": 1154,
      "train_loss": 0.0006928805248915321,
      "validation_loss": null
    },
    {
      "epoch": 1155,
      "train_loss": 0.0004976125880299757,
      "validation_loss": null
    },
    {
      "epoch": 1156,
      "train_loss": 0.0011471839485845219,
      "validation_loss": null
    },
    {
      "epoch": 1157,
      "train_loss": 0.000738615889632557,
      "validation_loss": null
    },
    {
      "epoch": 1158,
      "train_loss": 0.0006447158424028506,
      "validation_loss": null
    },
    {
      "epoch": 1159,
      "train_loss": 0.0005348875145298532,
      "validation_loss": null
    },
    {
      "epoch": 1160,
      "train_loss": 0.00030664741643704474,
      "validation_loss": null
    },
    {
      "epoch": 1161,
      "train_loss": 0.0001704090633817638,
      "validation_loss": null
    },
    {
      "epoch": 1162,
      "train_loss": 0.00020739627264750501,
      "validation_loss": null
    },
    {
      "epoch": 1163,
      "train_loss": 0.0001557413512879672,
      "validation_loss": null
    },
    {
      "epoch": 1164,
      "train_loss": 7.956235317073556e-05,
      "validation_loss": null
    },
    {
      "epoch": 1165,
      "train_loss": 7.78626498585153e-05,
      "validation_loss": null
    },
    {
      "epoch": 1166,
      "train_loss": 4.944997878434757e-05,
      "validation_loss": null
    },
    {
      "epoch": 1167,
      "train_loss": 5.008338505124104e-05,
      "validation_loss": null
    },
    {
      "epoch": 1168,
      "train_loss": 4.9094370398153034e-05,
      "validation_loss": null
    },
    {
      "epoch": 1169,
      "train_loss": 4.7176999133726997e-05,
      "validation_loss": null
    },
    {
      "epoch": 1170,
      "train_loss": 6.467891257165725e-05,
      "validation_loss": null
    },
    {
      "epoch": 1171,
      "train_loss": 0.00010906892202910967,
      "validation_loss": null
    },
    {
      "epoch": 1172,
      "train_loss": 0.00015231525564255813,
      "validation_loss": null
    },
    {
      "epoch": 1173,
      "train_loss": 0.0002521696814154792,
      "validation_loss": null
    },
    {
      "epoch": 1174,
      "train_loss": 0.00012433582014637068,
      "validation_loss": null
    },
    {
      "epoch": 1175,
      "train_loss": 7.742463276372291e-05,
      "validation_loss": null
    },
    {
      "epoch": 1176,
      "train_loss": 7.061971417796385e-05,
      "validation_loss": null
    },
    {
      "epoch": 1177,
      "train_loss": 0.00012306918183134662,
      "validation_loss": null
    },
    {
      "epoch": 1178,
      "train_loss": 0.00014488175980255214,
      "validation_loss": null
    },
    {
      "epoch": 1179,
      "train_loss": 0.00016765696798554726,
      "validation_loss": null
    },
    {
      "epoch": 1180,
      "train_loss": 0.00017600545834284276,
      "validation_loss": null
    },
    {
      "epoch": 1181,
      "train_loss": 7.333809521191547e-05,
      "validation_loss": null
    },
    {
      "epoch": 1182,
      "train_loss": 5.7018671618101914e-05,
      "validation_loss": null
    },
    {
      "epoch": 1183,
      "train_loss": 6.0148229143427066e-05,
      "validation_loss": null
    },
    {
      "epoch": 1184,
      "train_loss": 4.251165713261192e-05,
      "validation_loss": null
    },
    {
      "epoch": 1185,
      "train_loss": 5.186245309434728e-05,
      "validation_loss": null
    },
    {
      "epoch": 1186,
      "train_loss": 7.235618775464698e-05,
      "validation_loss": null
    },
    {
      "epoch": 1187,
      "train_loss": 6.769437944361319e-05,
      "validation_loss": null
    },
    {
      "epoch": 1188,
      "train_loss": 8.982215432398435e-05,
      "validation_loss": null
    },
    {
      "epoch": 1189,
      "train_loss": 5.571849639333474e-05,
      "validation_loss": null
    },
    {
      "epoch": 1190,
      "train_loss": 3.9893335471990416e-05,
      "validation_loss": null
    },
    {
      "epoch": 1191,
      "train_loss": 3.744356743279948e-05,
      "validation_loss": null
    },
    {
      "epoch": 1192,
      "train_loss": 4.698545995375348e-05,
      "validation_loss": null
    },
    {
      "epoch": 1193,
      "train_loss": 4.514971058395329e-05,
      "validation_loss": null
    },
    {
      "epoch": 1194,
      "train_loss": 3.794243336111928e-05,
      "validation_loss": null
    },
    {
      "epoch": 1195,
      "train_loss": 3.1527786632068455e-05,
      "validation_loss": null
    },
    {
      "epoch": 1196,
      "train_loss": 4.75262636933217e-05,
      "validation_loss": null
    },
    {
      "epoch": 1197,
      "train_loss": 8.23032019575799e-05,
      "validation_loss": null
    },
    {
      "epoch": 1198,
      "train_loss": 8.118411440389334e-05,
      "validation_loss": null
    },
    {
      "epoch": 1199,
      "train_loss": 5.405538791415488e-05,
      "validation_loss": null
    },
    {
      "epoch": 1200,
      "train_loss": 4.4930745288082915e-05,
      "validation_loss": null
    },
    {
      "epoch": 1201,
      "train_loss": 5.5368724942026245e-05,
      "validation_loss": null
    },
    {
      "epoch": 1202,
      "train_loss": 4.2912960149503e-05,
      "validation_loss": null
    },
    {
      "epoch": 1203,
      "train_loss": 5.6703080190345645e-05,
      "validation_loss": null
    },
    {
      "epoch": 1204,
      "train_loss": 5.4529413723179867e-05,
      "validation_loss": null
    },
    {
      "epoch": 1205,
      "train_loss": 6.383506539956822e-05,
      "validation_loss": null
    },
    {
      "epoch": 1206,
      "train_loss": 5.394928666646592e-05,
      "validation_loss": null
    },
    {
      "epoch": 1207,
      "train_loss": 5.8901052398141474e-05,
      "validation_loss": null
    },
    {
      "epoch": 1208,
      "train_loss": 8.072773925960064e-05,
      "validation_loss": null
    },
    {
      "epoch": 1209,
      "train_loss": 8.489986834724227e-05,
      "validation_loss": null
    },
    {
      "epoch": 1210,
      "train_loss": 6.7982445923715e-05,
      "validation_loss": null
    },
    {
      "epoch": 1211,
      "train_loss": 0.00011279778684385949,
      "validation_loss": null
    },
    {
      "epoch": 1212,
      "train_loss": 0.000100460877648503,
      "validation_loss": null
    },
    {
      "epoch": 1213,
      "train_loss": 0.00014829151162606044,
      "validation_loss": null
    },
    {
      "epoch": 1214,
      "train_loss": 0.00017058167092424506,
      "validation_loss": null
    },
    {
      "epoch": 1215,
      "train_loss": 0.00044647870203738823,
      "validation_loss": null
    },
    {
      "epoch": 1216,
      "train_loss": 0.0002496647017930324,
      "validation_loss": null
    },
    {
      "epoch": 1217,
      "train_loss": 0.0001117443167054767,
      "validation_loss": null
    },
    {
      "epoch": 1218,
      "train_loss": 8.478190850130179e-05,
      "validation_loss": null
    },
    {
      "epoch": 1219,
      "train_loss": 6.376814457730183e-05,
      "validation_loss": null
    },
    {
      "epoch": 1220,
      "train_loss": 9.326505055974444e-05,
      "validation_loss": null
    },
    {
      "epoch": 1221,
      "train_loss": 0.000141945530130114,
      "validation_loss": null
    },
    {
      "epoch": 1222,
      "train_loss": 0.0002621544748156642,
      "validation_loss": null
    },
    {
      "epoch": 1223,
      "train_loss": 0.00028881796525739547,
      "validation_loss": null
    },
    {
      "epoch": 1224,
      "train_loss": 0.000379826514593636,
      "validation_loss": null
    },
    {
      "epoch": 1225,
      "train_loss": 0.00023528956646461866,
      "validation_loss": null
    },
    {
      "epoch": 1226,
      "train_loss": 0.0004216276995268547,
      "validation_loss": null
    },
    {
      "epoch": 1227,
      "train_loss": 0.0004113719397638407,
      "validation_loss": null
    },
    {
      "epoch": 1228,
      "train_loss": 0.0004484719182882044,
      "validation_loss": null
    },
    {
      "epoch": 1229,
      "train_loss": 0.00048067539723383053,
      "validation_loss": null
    },
    {
      "epoch": 1230,
      "train_loss": 0.0007063124438799503,
      "validation_loss": null
    },
    {
      "epoch": 1231,
      "train_loss": 0.001714536884618509,
      "validation_loss": null
    },
    {
      "epoch": 1232,
      "train_loss": 0.010927028359017439,
      "validation_loss": null
    },
    {
      "epoch": 1233,
      "train_loss": 0.016049439190990396,
      "validation_loss": null
    },
    {
      "epoch": 1234,
      "train_loss": 0.16543079188300502,
      "validation_loss": null
    },
    {
      "epoch": 1235,
      "train_loss": 0.13478607146276367,
      "validation_loss": null
    },
    {
      "epoch": 1236,
      "train_loss": 0.04465688392519951,
      "validation_loss": null
    },
    {
      "epoch": 1237,
      "train_loss": 0.023577720237274963,
      "validation_loss": null
    },
    {
      "epoch": 1238,
      "train_loss": 0.011462896389679776,
      "validation_loss": null
    },
    {
      "epoch": 1239,
      "train_loss": 0.0058214509756200844,
      "validation_loss": null
    },
    {
      "epoch": 1240,
      "train_loss": 0.004028524660194914,
      "validation_loss": null
    },
    {
      "epoch": 1241,
      "train_loss": 0.0030545727867219183,
      "validation_loss": null
    },
    {
      "epoch": 1242,
      "train_loss": 0.0021127651238607038,
      "validation_loss": null
    },
    {
      "epoch": 1243,
      "train_loss": 0.0017160770043523775,
      "validation_loss": null
    },
    {
      "epoch": 1244,
      "train_loss": 0.0015214814468183452,
      "validation_loss": null
    },
    {
      "epoch": 1245,
      "train_loss": 0.001334780503788756,
      "validation_loss": null
    },
    {
      "epoch": 1246,
      "train_loss": 0.0012208858129775359,
      "validation_loss": null
    },
    {
      "epoch": 1247,
      "train_loss": 0.0010876721935346723,
      "validation_loss": null
    },
    {
      "epoch": 1248,
      "train_loss": 0.0009725041001931661,
      "validation_loss": null
    },
    {
      "epoch": 1249,
      "train_loss": 0.0009054489394960304,
      "validation_loss": null
    },
    {
      "epoch": 1250,
      "train_loss": 0.0009051554694047405,
      "validation_loss": null
    },
    {
      "epoch": 1251,
      "train_loss": 0.0007825538761810296,
      "validation_loss": null
    },
    {
      "epoch": 1252,
      "train_loss": 0.0007614328318999873,
      "validation_loss": null
    },
    {
      "epoch": 1253,
      "train_loss": 0.0007055227955182394,
      "validation_loss": null
    },
    {
      "epoch": 1254,
      "train_loss": 0.0006685226431323423,
      "validation_loss": null
    },
    {
      "epoch": 1255,
      "train_loss": 0.000623403673267199,
      "validation_loss": null
    },
    {
      "epoch": 1256,
      "train_loss": 0.0005727484919286022,
      "validation_loss": null
    },
    {
      "epoch": 1257,
      "train_loss": 0.0005527944692100087,
      "validation_loss": null
    },
    {
      "epoch": 1258,
      "train_loss": 0.0005141437448199011,
      "validation_loss": null
    },
    {
      "epoch": 1259,
      "train_loss": 0.0004636931375393437,
      "validation_loss": null
    },
    {
      "epoch": 1260,
      "train_loss": 0.0004528868076805439,
      "validation_loss": null
    },
    {
      "epoch": 1261,
      "train_loss": 0.0004396236941425337,
      "validation_loss": null
    },
    {
      "epoch": 1262,
      "train_loss": 0.0004227484557001541,
      "validation_loss": null
    },
    {
      "epoch": 1263,
      "train_loss": 0.0003936812313946171,
      "validation_loss": null
    },
    {
      "epoch": 1264,
      "train_loss": 0.00037149418009600293,
      "validation_loss": null
    },
    {
      "epoch": 1265,
      "train_loss": 0.0003508056462224987,
      "validation_loss": null
    },
    {
      "epoch": 1266,
      "train_loss": 0.0003494378405674878,
      "validation_loss": null
    },
    {
      "epoch": 1267,
      "train_loss": 0.00037246006993680366,
      "validation_loss": null
    },
    {
      "epoch": 1268,
      "train_loss": 0.00032933529615143523,
      "validation_loss": null
    },
    {
      "epoch": 1269,
      "train_loss": 0.0003087677533686575,
      "validation_loss": null
    },
    {
      "epoch": 1270,
      "train_loss": 0.000307972424909369,
      "validation_loss": null
    },
    {
      "epoch": 1271,
      "train_loss": 0.00028113302929947775,
      "validation_loss": null
    },
    {
      "epoch": 1272,
      "train_loss": 0.0002718913504698624,
      "validation_loss": null
    },
    {
      "epoch": 1273,
      "train_loss": 0.00027442313795391884,
      "validation_loss": null
    },
    {
      "epoch": 1274,
      "train_loss": 0.0002636995227981566,
      "validation_loss": null
    },
    {
      "epoch": 1275,
      "train_loss": 0.0002544344056837468,
      "validation_loss": null
    },
    {
      "epoch": 1276,
      "train_loss": 0.0002398690558038652,
      "validation_loss": null
    },
    {
      "epoch": 1277,
      "train_loss": 0.0002362052441134842,
      "validation_loss": null
    },
    {
      "epoch": 1278,
      "train_loss": 0.00023236534939820154,
      "validation_loss": null
    },
    {
      "epoch": 1279,
      "train_loss": 0.00023027720453683287,
      "validation_loss": null
    },
    {
      "epoch": 1280,
      "train_loss": 0.00021641232564838396,
      "validation_loss": null
    },
    {
      "epoch": 1281,
      "train_loss": 0.00019762681929730915,
      "validation_loss": null
    },
    {
      "epoch": 1282,
      "train_loss": 0.0002041638055945643,
      "validation_loss": null
    },
    {
      "epoch": 1283,
      "train_loss": 0.0001939278315856225,
      "validation_loss": null
    },
    {
      "epoch": 1284,
      "train_loss": 0.0001915638357685465,
      "validation_loss": null
    },
    {
      "epoch": 1285,
      "train_loss": 0.0001997597315089984,
      "validation_loss": null
    },
    {
      "epoch": 1286,
      "train_loss": 0.0001816518375158517,
      "validation_loss": null
    },
    {
      "epoch": 1287,
      "train_loss": 0.00018056224022681514,
      "validation_loss": null
    },
    {
      "epoch": 1288,
      "train_loss": 0.00018216439477530203,
      "validation_loss": null
    },
    {
      "epoch": 1289,
      "train_loss": 0.0001823186369923254,
      "validation_loss": null
    },
    {
      "epoch": 1290,
      "train_loss": 0.00016570146868212355,
      "validation_loss": null
    },
    {
      "epoch": 1291,
      "train_loss": 0.00016414163433687968,
      "validation_loss": null
    },
    {
      "epoch": 1292,
      "train_loss": 0.00015886906799601598,
      "validation_loss": null
    },
    {
      "epoch": 1293,
      "train_loss": 0.0001549571712126231,
      "validation_loss": null
    },
    {
      "epoch": 1294,
      "train_loss": 0.00014295953345329812,
      "validation_loss": null
    },
    {
      "epoch": 1295,
      "train_loss": 0.00014660176141963652,
      "validation_loss": null
    },
    {
      "epoch": 1296,
      "train_loss": 0.00014145019991297481,
      "validation_loss": null
    },
    {
      "epoch": 1297,
      "train_loss": 0.00013735142096670138,
      "validation_loss": null
    },
    {
      "epoch": 1298,
      "train_loss": 0.00013336475886818435,
      "validation_loss": null
    },
    {
      "epoch": 1299,
      "train_loss": 0.00013985253139657693,
      "validation_loss": null
    },
    {
      "epoch": 1300,
      "train_loss": 0.00012995908845267777,
      "validation_loss": null
    },
    {
      "epoch": 1301,
      "train_loss": 0.00012383540969393734,
      "validation_loss": null
    },
    {
      "epoch": 1302,
      "train_loss": 0.00012181271045442877,
      "validation_loss": null
    },
    {
      "epoch": 1303,
      "train_loss": 0.0001201275364211243,
      "validation_loss": null
    },
    {
      "epoch": 1304,
      "train_loss": 0.00011607838031421934,
      "validation_loss": null
    },
    {
      "epoch": 1305,
      "train_loss": 0.00011505067797972717,
      "validation_loss": null
    },
    {
      "epoch": 1306,
      "train_loss": 0.00011000871018040925,
      "validation_loss": null
    },
    {
      "epoch": 1307,
      "train_loss": 0.00011143952077772055,
      "validation_loss": null
    },
    {
      "epoch": 1308,
      "train_loss": 0.00010789221899661546,
      "validation_loss": null
    },
    {
      "epoch": 1309,
      "train_loss": 0.0001200454700867542,
      "validation_loss": null
    },
    {
      "epoch": 1310,
      "train_loss": 0.00011110454862419929,
      "validation_loss": null
    },
    {
      "epoch": 1311,
      "train_loss": 0.00010781810043327924,
      "validation_loss": null
    },
    {
      "epoch": 1312,
      "train_loss": 0.00011088230288199459,
      "validation_loss": null
    },
    {
      "epoch": 1313,
      "train_loss": 0.00011375695774202338,
      "validation_loss": null
    },
    {
      "epoch": 1314,
      "train_loss": 0.0001259136010452898,
      "validation_loss": null
    },
    {
      "epoch": 1315,
      "train_loss": 0.00011968177002725295,
      "validation_loss": null
    },
    {
      "epoch": 1316,
      "train_loss": 0.00010096694071156283,
      "validation_loss": null
    },
    {
      "epoch": 1317,
      "train_loss": 9.268718915538759e-05,
      "validation_loss": null
    },
    {
      "epoch": 1318,
      "train_loss": 9.254292187203343e-05,
      "validation_loss": null
    },
    {
      "epoch": 1319,
      "train_loss": 9.410862629819248e-05,
      "validation_loss": null
    },
    {
      "epoch": 1320,
      "train_loss": 8.814316485465194e-05,
      "validation_loss": null
    },
    {
      "epoch": 1321,
      "train_loss": 8.909224076584603e-05,
      "validation_loss": null
    },
    {
      "epoch": 1322,
      "train_loss": 8.939115933672938e-05,
      "validation_loss": null
    },
    {
      "epoch": 1323,
      "train_loss": 8.373381731669522e-05,
      "validation_loss": null
    },
    {
      "epoch": 1324,
      "train_loss": 8.40020665135752e-05,
      "validation_loss": null
    },
    {
      "epoch": 1325,
      "train_loss": 8.620659516762114e-05,
      "validation_loss": null
    },
    {
      "epoch": 1326,
      "train_loss": 8.257529892337818e-05,
      "validation_loss": null
    },
    {
      "epoch": 1327,
      "train_loss": 8.145773042795352e-05,
      "validation_loss": null
    },
    {
      "epoch": 1328,
      "train_loss": 8.394934532892269e-05,
      "validation_loss": null
    },
    {
      "epoch": 1329,
      "train_loss": 8.827107174309074e-05,
      "validation_loss": null
    },
    {
      "epoch": 1330,
      "train_loss": 8.041509419045824e-05,
      "validation_loss": null
    },
    {
      "epoch": 1331,
      "train_loss": 7.475028057039405e-05,
      "validation_loss": null
    },
    {
      "epoch": 1332,
      "train_loss": 7.39916928675181e-05,
      "validation_loss": null
    },
    {
      "epoch": 1333,
      "train_loss": 7.949863437614922e-05,
      "validation_loss": null
    },
    {
      "epoch": 1334,
      "train_loss": 8.049051312910806e-05,
      "validation_loss": null
    },
    {
      "epoch": 1335,
      "train_loss": 7.702334066076826e-05,
      "validation_loss": null
    },
    {
      "epoch": 1336,
      "train_loss": 7.507950390896035e-05,
      "validation_loss": null
    },
    {
      "epoch": 1337,
      "train_loss": 7.302810465464265e-05,
      "validation_loss": null
    },
    {
      "epoch": 1338,
      "train_loss": 7.14883386131583e-05,
      "validation_loss": null
    },
    {
      "epoch": 1339,
      "train_loss": 7.002940037637018e-05,
      "validation_loss": null
    },
    {
      "epoch": 1340,
      "train_loss": 7.400560409425654e-05,
      "validation_loss": null
    },
    {
      "epoch": 1341,
      "train_loss": 7.543523315689526e-05,
      "validation_loss": null
    },
    {
      "epoch": 1342,
      "train_loss": 6.993111108183964e-05,
      "validation_loss": null
    },
    {
      "epoch": 1343,
      "train_loss": 6.767807361837994e-05,
      "validation_loss": null
    },
    {
      "epoch": 1344,
      "train_loss": 6.5604943099768e-05,
      "validation_loss": null
    },
    {
      "epoch": 1345,
      "train_loss": 6.600195370588658e-05,
      "validation_loss": null
    },
    {
      "epoch": 1346,
      "train_loss": 6.583919498047585e-05,
      "validation_loss": null
    },
    {
      "epoch": 1347,
      "train_loss": 6.498051880043931e-05,
      "validation_loss": null
    },
    {
      "epoch": 1348,
      "train_loss": 6.510941117691497e-05,
      "validation_loss": null
    },
    {
      "epoch": 1349,
      "train_loss": 6.596444735704508e-05,
      "validation_loss": null
    },
    {
      "epoch": 1350,
      "train_loss": 6.111920487860011e-05,
      "validation_loss": null
    },
    {
      "epoch": 1351,
      "train_loss": 6.175495986503342e-05,
      "validation_loss": null
    },
    {
      "epoch": 1352,
      "train_loss": 6.086167640104476e-05,
      "validation_loss": null
    },
    {
      "epoch": 1353,
      "train_loss": 5.908612891087412e-05,
      "validation_loss": null
    },
    {
      "epoch": 1354,
      "train_loss": 5.9468471590662375e-05,
      "validation_loss": null
    },
    {
      "epoch": 1355,
      "train_loss": 5.8592570389414e-05,
      "validation_loss": null
    },
    {
      "epoch": 1356,
      "train_loss": 5.8429016159304105e-05,
      "validation_loss": null
    },
    {
      "epoch": 1357,
      "train_loss": 5.723075688971827e-05,
      "validation_loss": null
    },
    {
      "epoch": 1358,
      "train_loss": 5.9431644735418056e-05,
      "validation_loss": null
    },
    {
      "epoch": 1359,
      "train_loss": 5.992601371771242e-05,
      "validation_loss": null
    },
    {
      "epoch": 1360,
      "train_loss": 5.91581067257923e-05,
      "validation_loss": null
    },
    {
      "epoch": 1361,
      "train_loss": 6.276527023550848e-05,
      "validation_loss": null
    },
    {
      "epoch": 1362,
      "train_loss": 5.783710730611347e-05,
      "validation_loss": null
    },
    {
      "epoch": 1363,
      "train_loss": 5.574490488571529e-05,
      "validation_loss": null
    },
    {
      "epoch": 1364,
      "train_loss": 5.283396280396523e-05,
      "validation_loss": null
    },
    {
      "epoch": 1365,
      "train_loss": 5.38674539711792e-05,
      "validation_loss": null
    },
    {
      "epoch": 1366,
      "train_loss": 5.237454024609178e-05,
      "validation_loss": null
    },
    {
      "epoch": 1367,
      "train_loss": 5.343393624773146e-05,
      "validation_loss": null
    },
    {
      "epoch": 1368,
      "train_loss": 5.2947451169731925e-05,
      "validation_loss": null
    },
    {
      "epoch": 1369,
      "train_loss": 5.1780134930999746e-05,
      "validation_loss": null
    },
    {
      "epoch": 1370,
      "train_loss": 5.437377533073434e-05,
      "validation_loss": null
    },
    {
      "epoch": 1371,
      "train_loss": 5.317495136599367e-05,
      "validation_loss": null
    },
    {
      "epoch": 1372,
      "train_loss": 5.019525552698825e-05,
      "validation_loss": null
    },
    {
      "epoch": 1373,
      "train_loss": 5.263267707454765e-05,
      "validation_loss": null
    },
    {
      "epoch": 1374,
      "train_loss": 4.863420609682281e-05,
      "validation_loss": null
    },
    {
      "epoch": 1375,
      "train_loss": 5.0291347886539166e-05,
      "validation_loss": null
    },
    {
      "epoch": 1376,
      "train_loss": 5.103763477462861e-05,
      "validation_loss": null
    },
    {
      "epoch": 1377,
      "train_loss": 4.884923247219477e-05,
      "validation_loss": null
    },
    {
      "epoch": 1378,
      "train_loss": 5.3616388565084584e-05,
      "validation_loss": null
    },
    {
      "epoch": 1379,
      "train_loss": 5.796973750370348e-05,
      "validation_loss": null
    },
    {
      "epoch": 1380,
      "train_loss": 5.443694357786121e-05,
      "validation_loss": null
    },
    {
      "epoch": 1381,
      "train_loss": 4.999231572987305e-05,
      "validation_loss": null
    },
    {
      "epoch": 1382,
      "train_loss": 5.053681888966821e-05,
      "validation_loss": null
    },
    {
      "epoch": 1383,
      "train_loss": 5.1111958681657496e-05,
      "validation_loss": null
    },
    {
      "epoch": 1384,
      "train_loss": 4.618124674177832e-05,
      "validation_loss": null
    },
    {
      "epoch": 1385,
      "train_loss": 4.830783614630086e-05,
      "validation_loss": null
    },
    {
      "epoch": 1386,
      "train_loss": 4.7619636461604387e-05,
      "validation_loss": null
    },
    {
      "epoch": 1387,
      "train_loss": 4.721288577355962e-05,
      "validation_loss": null
    },
    {
      "epoch": 1388,
      "train_loss": 4.536002799997934e-05,
      "validation_loss": null
    },
    {
      "epoch": 1389,
      "train_loss": 4.663603572731113e-05,
      "validation_loss": null
    },
    {
      "epoch": 1390,
      "train_loss": 4.809213553218999e-05,
      "validation_loss": null
    },
    {
      "epoch": 1391,
      "train_loss": 4.943113728788578e-05,
      "validation_loss": null
    },
    {
      "epoch": 1392,
      "train_loss": 4.6671531890751794e-05,
      "validation_loss": null
    },
    {
      "epoch": 1393,
      "train_loss": 4.395001491906846e-05,
      "validation_loss": null
    },
    {
      "epoch": 1394,
      "train_loss": 4.4416805596685866e-05,
      "validation_loss": null
    },
    {
      "epoch": 1395,
      "train_loss": 4.41197326937173e-05,
      "validation_loss": null
    },
    {
      "epoch": 1396,
      "train_loss": 4.470256559822398e-05,
      "validation_loss": null
    },
    {
      "epoch": 1397,
      "train_loss": 4.469075510213669e-05,
      "validation_loss": null
    },
    {
      "epoch": 1398,
      "train_loss": 4.587965546913134e-05,
      "validation_loss": null
    },
    {
      "epoch": 1399,
      "train_loss": 4.578662588351613e-05,
      "validation_loss": null
    },
    {
      "epoch": 1400,
      "train_loss": 4.2385899076988506e-05,
      "validation_loss": null
    },
    {
      "epoch": 1401,
      "train_loss": 4.4985390559304506e-05,
      "validation_loss": null
    },
    {
      "epoch": 1402,
      "train_loss": 4.261846515873913e-05,
      "validation_loss": null
    },
    {
      "epoch": 1403,
      "train_loss": 4.3483477687307946e-05,
      "validation_loss": null
    },
    {
      "epoch": 1404,
      "train_loss": 4.0916134619490345e-05,
      "validation_loss": null
    },
    {
      "epoch": 1405,
      "train_loss": 4.235699452692643e-05,
      "validation_loss": null
    },
    {
      "epoch": 1406,
      "train_loss": 4.114345817167001e-05,
      "validation_loss": null
    },
    {
      "epoch": 1407,
      "train_loss": 4.3493549229525444e-05,
      "validation_loss": null
    },
    {
      "epoch": 1408,
      "train_loss": 4.831401525330471e-05,
      "validation_loss": null
    },
    {
      "epoch": 1409,
      "train_loss": 4.688486559542879e-05,
      "validation_loss": null
    },
    {
      "epoch": 1410,
      "train_loss": 3.879138764912366e-05,
      "validation_loss": null
    },
    {
      "epoch": 1411,
      "train_loss": 3.8865562196911517e-05,
      "validation_loss": null
    },
    {
      "epoch": 1412,
      "train_loss": 3.678217257806359e-05,
      "validation_loss": null
    },
    {
      "epoch": 1413,
      "train_loss": 3.810700097043688e-05,
      "validation_loss": null
    },
    {
      "epoch": 1414,
      "train_loss": 3.8250885053356695e-05,
      "validation_loss": null
    },
    {
      "epoch": 1415,
      "train_loss": 3.780272891162895e-05,
      "validation_loss": null
    },
    {
      "epoch": 1416,
      "train_loss": 3.7275943112844187e-05,
      "validation_loss": null
    },
    {
      "epoch": 1417,
      "train_loss": 3.729450731447691e-05,
      "validation_loss": null
    },
    {
      "epoch": 1418,
      "train_loss": 3.798821166532839e-05,
      "validation_loss": null
    },
    {
      "epoch": 1419,
      "train_loss": 4.1602837073797775e-05,
      "validation_loss": null
    },
    {
      "epoch": 1420,
      "train_loss": 4.1016915121064005e-05,
      "validation_loss": null
    },
    {
      "epoch": 1421,
      "train_loss": 5.268526688774323e-05,
      "validation_loss": null
    },
    {
      "epoch": 1422,
      "train_loss": 4.192667640583952e-05,
      "validation_loss": null
    },
    {
      "epoch": 1423,
      "train_loss": 3.878607033887723e-05,
      "validation_loss": null
    },
    {
      "epoch": 1424,
      "train_loss": 4.163800339382659e-05,
      "validation_loss": null
    },
    {
      "epoch": 1425,
      "train_loss": 4.037379929084434e-05,
      "validation_loss": null
    },
    {
      "epoch": 1426,
      "train_loss": 3.971983339094246e-05,
      "validation_loss": null
    },
    {
      "epoch": 1427,
      "train_loss": 3.92946199604517e-05,
      "validation_loss": null
    },
    {
      "epoch": 1428,
      "train_loss": 3.6904084279538234e-05,
      "validation_loss": null
    },
    {
      "epoch": 1429,
      "train_loss": 3.89845311777511e-05,
      "validation_loss": null
    },
    {
      "epoch": 1430,
      "train_loss": 3.926438775023497e-05,
      "validation_loss": null
    },
    {
      "epoch": 1431,
      "train_loss": 3.571445449779276e-05,
      "validation_loss": null
    },
    {
      "epoch": 1432,
      "train_loss": 3.5796510322446316e-05,
      "validation_loss": null
    },
    {
      "epoch": 1433,
      "train_loss": 3.404222459519385e-05,
      "validation_loss": null
    },
    {
      "epoch": 1434,
      "train_loss": 3.4005184716079384e-05,
      "validation_loss": null
    },
    {
      "epoch": 1435,
      "train_loss": 3.460984084490014e-05,
      "validation_loss": null
    },
    {
      "epoch": 1436,
      "train_loss": 3.623236403250808e-05,
      "validation_loss": null
    },
    {
      "epoch": 1437,
      "train_loss": 3.328366680458809e-05,
      "validation_loss": null
    },
    {
      "epoch": 1438,
      "train_loss": 3.309093052747711e-05,
      "validation_loss": null
    },
    {
      "epoch": 1439,
      "train_loss": 3.388970354636614e-05,
      "validation_loss": null
    },
    {
      "epoch": 1440,
      "train_loss": 3.191974974470213e-05,
      "validation_loss": null
    },
    {
      "epoch": 1441,
      "train_loss": 3.2576084575137225e-05,
      "validation_loss": null
    },
    {
      "epoch": 1442,
      "train_loss": 3.2119318146012825e-05,
      "validation_loss": null
    },
    {
      "epoch": 1443,
      "train_loss": 3.2478203744782754e-05,
      "validation_loss": null
    },
    {
      "epoch": 1444,
      "train_loss": 3.378911707032886e-05,
      "validation_loss": null
    },
    {
      "epoch": 1445,
      "train_loss": 3.342727419900863e-05,
      "validation_loss": null
    },
    {
      "epoch": 1446,
      "train_loss": 3.219392096475026e-05,
      "validation_loss": null
    },
    {
      "epoch": 1447,
      "train_loss": 3.229134280344523e-05,
      "validation_loss": null
    },
    {
      "epoch": 1448,
      "train_loss": 3.138232770854504e-05,
      "validation_loss": null
    },
    {
      "epoch": 1449,
      "train_loss": 3.233779797382239e-05,
      "validation_loss": null
    },
    {
      "epoch": 1450,
      "train_loss": 3.282773145151118e-05,
      "validation_loss": null
    },
    {
      "epoch": 1451,
      "train_loss": 3.2867233029618445e-05,
      "validation_loss": null
    },
    {
      "epoch": 1452,
      "train_loss": 3.064015032173807e-05,
      "validation_loss": null
    },
    {
      "epoch": 1453,
      "train_loss": 3.16066717560817e-05,
      "validation_loss": null
    },
    {
      "epoch": 1454,
      "train_loss": 3.1469094513643846e-05,
      "validation_loss": null
    },
    {
      "epoch": 1455,
      "train_loss": 2.938950698686919e-05,
      "validation_loss": null
    },
    {
      "epoch": 1456,
      "train_loss": 3.515294499973404e-05,
      "validation_loss": null
    },
    {
      "epoch": 1457,
      "train_loss": 3.421376156135617e-05,
      "validation_loss": null
    },
    {
      "epoch": 1458,
      "train_loss": 3.550924298502246e-05,
      "validation_loss": null
    },
    {
      "epoch": 1459,
      "train_loss": 4.056015232991841e-05,
      "validation_loss": null
    },
    {
      "epoch": 1460,
      "train_loss": 3.9676476363739413e-05,
      "validation_loss": null
    },
    {
      "epoch": 1461,
      "train_loss": 3.643778856914853e-05,
      "validation_loss": null
    },
    {
      "epoch": 1462,
      "train_loss": 3.3776074309975636e-05,
      "validation_loss": null
    },
    {
      "epoch": 1463,
      "train_loss": 3.002273499280111e-05,
      "validation_loss": null
    },
    {
      "epoch": 1464,
      "train_loss": 2.8753856491271614e-05,
      "validation_loss": null
    },
    {
      "epoch": 1465,
      "train_loss": 3.0787037151943274e-05,
      "validation_loss": null
    },
    {
      "epoch": 1466,
      "train_loss": 3.150436916459714e-05,
      "validation_loss": null
    },
    {
      "epoch": 1467,
      "train_loss": 3.573915374646377e-05,
      "validation_loss": null
    },
    {
      "epoch": 1468,
      "train_loss": 4.119425648241304e-05,
      "validation_loss": null
    },
    {
      "epoch": 1469,
      "train_loss": 3.336021634620718e-05,
      "validation_loss": null
    },
    {
      "epoch": 1470,
      "train_loss": 3.180217956671388e-05,
      "validation_loss": null
    },
    {
      "epoch": 1471,
      "train_loss": 3.867631413514674e-05,
      "validation_loss": null
    },
    {
      "epoch": 1472,
      "train_loss": 3.149722377000014e-05,
      "validation_loss": null
    },
    {
      "epoch": 1473,
      "train_loss": 3.1270820828568605e-05,
      "validation_loss": null
    },
    {
      "epoch": 1474,
      "train_loss": 2.9713694958546614e-05,
      "validation_loss": null
    },
    {
      "epoch": 1475,
      "train_loss": 2.9512645495641562e-05,
      "validation_loss": null
    },
    {
      "epoch": 1476,
      "train_loss": 3.0419351484548923e-05,
      "validation_loss": null
    },
    {
      "epoch": 1477,
      "train_loss": 3.0298104522646303e-05,
      "validation_loss": null
    },
    {
      "epoch": 1478,
      "train_loss": 2.6647826517647547e-05,
      "validation_loss": null
    },
    {
      "epoch": 1479,
      "train_loss": 3.045086204009648e-05,
      "validation_loss": null
    },
    {
      "epoch": 1480,
      "train_loss": 2.626292775352744e-05,
      "validation_loss": null
    },
    {
      "epoch": 1481,
      "train_loss": 2.8616613538664144e-05,
      "validation_loss": null
    },
    {
      "epoch": 1482,
      "train_loss": 2.589829859466085e-05,
      "validation_loss": null
    },
    {
      "epoch": 1483,
      "train_loss": 2.6331374910013336e-05,
      "validation_loss": null
    },
    {
      "epoch": 1484,
      "train_loss": 3.1577114795153546e-05,
      "validation_loss": null
    },
    {
      "epoch": 1485,
      "train_loss": 3.0353306202515443e-05,
      "validation_loss": null
    },
    {
      "epoch": 1486,
      "train_loss": 3.0485724588894907e-05,
      "validation_loss": null
    },
    {
      "epoch": 1487,
      "train_loss": 3.46727111819847e-05,
      "validation_loss": null
    },
    {
      "epoch": 1488,
      "train_loss": 3.482769190870588e-05,
      "validation_loss": null
    },
    {
      "epoch": 1489,
      "train_loss": 3.0926209243868165e-05,
      "validation_loss": null
    },
    {
      "epoch": 1490,
      "train_loss": 3.273876791354269e-05,
      "validation_loss": null
    },
    {
      "epoch": 1491,
      "train_loss": 3.307261653794235e-05,
      "validation_loss": null
    },
    {
      "epoch": 1492,
      "train_loss": 2.806501935184416e-05,
      "validation_loss": null
    },
    {
      "epoch": 1493,
      "train_loss": 2.722785514682376e-05,
      "validation_loss": null
    },
    {
      "epoch": 1494,
      "train_loss": 2.581954969274294e-05,
      "validation_loss": null
    },
    {
      "epoch": 1495,
      "train_loss": 2.6102732388406163e-05,
      "validation_loss": null
    },
    {
      "epoch": 1496,
      "train_loss": 2.6842077002382008e-05,
      "validation_loss": null
    },
    {
      "epoch": 1497,
      "train_loss": 2.8397789921857315e-05,
      "validation_loss": null
    },
    {
      "epoch": 1498,
      "train_loss": 2.653372333346245e-05,
      "validation_loss": null
    },
    {
      "epoch": 1499,
      "train_loss": 2.534915175702837e-05,
      "validation_loss": null
    },
    {
      "epoch": 1500,
      "train_loss": 2.9942589713351077e-05,
      "validation_loss": null
    },
    {
      "epoch": 1501,
      "train_loss": 2.847845381539729e-05,
      "validation_loss": null
    },
    {
      "epoch": 1502,
      "train_loss": 2.4688951574110735e-05,
      "validation_loss": null
    },
    {
      "epoch": 1503,
      "train_loss": 2.279069061235835e-05,
      "validation_loss": null
    },
    {
      "epoch": 1504,
      "train_loss": 2.370060418696893e-05,
      "validation_loss": null
    },
    {
      "epoch": 1505,
      "train_loss": 2.499268521205522e-05,
      "validation_loss": null
    },
    {
      "epoch": 1506,
      "train_loss": 2.9975373965700986e-05,
      "validation_loss": null
    },
    {
      "epoch": 1507,
      "train_loss": 2.632651335766746e-05,
      "validation_loss": null
    },
    {
      "epoch": 1508,
      "train_loss": 2.5258350231322562e-05,
      "validation_loss": null
    },
    {
      "epoch": 1509,
      "train_loss": 3.019041893518685e-05,
      "validation_loss": null
    },
    {
      "epoch": 1510,
      "train_loss": 2.5563410923091902e-05,
      "validation_loss": null
    },
    {
      "epoch": 1511,
      "train_loss": 2.5383802696726183e-05,
      "validation_loss": null
    },
    {
      "epoch": 1512,
      "train_loss": 3.549746300753517e-05,
      "validation_loss": null
    },
    {
      "epoch": 1513,
      "train_loss": 3.9071586595835266e-05,
      "validation_loss": null
    },
    {
      "epoch": 1514,
      "train_loss": 3.7271311561602895e-05,
      "validation_loss": null
    },
    {
      "epoch": 1515,
      "train_loss": 3.1852360229499434e-05,
      "validation_loss": null
    },
    {
      "epoch": 1516,
      "train_loss": 2.6362485894221914e-05,
      "validation_loss": null
    },
    {
      "epoch": 1517,
      "train_loss": 2.6507745335240746e-05,
      "validation_loss": null
    },
    {
      "epoch": 1518,
      "train_loss": 2.3477279784047925e-05,
      "validation_loss": null
    },
    {
      "epoch": 1519,
      "train_loss": 2.293517678077074e-05,
      "validation_loss": null
    },
    {
      "epoch": 1520,
      "train_loss": 2.5784006235577788e-05,
      "validation_loss": null
    },
    {
      "epoch": 1521,
      "train_loss": 2.630055738942853e-05,
      "validation_loss": null
    },
    {
      "epoch": 1522,
      "train_loss": 2.4589037315713034e-05,
      "validation_loss": null
    },
    {
      "epoch": 1523,
      "train_loss": 2.439030948961671e-05,
      "validation_loss": null
    },
    {
      "epoch": 1524,
      "train_loss": 3.09230783588848e-05,
      "validation_loss": null
    },
    {
      "epoch": 1525,
      "train_loss": 3.0363494362164703e-05,
      "validation_loss": null
    },
    {
      "epoch": 1526,
      "train_loss": 2.5827747877984722e-05,
      "validation_loss": null
    },
    {
      "epoch": 1527,
      "train_loss": 3.663664473505277e-05,
      "validation_loss": null
    },
    {
      "epoch": 1528,
      "train_loss": 3.648742818364149e-05,
      "validation_loss": null
    },
    {
      "epoch": 1529,
      "train_loss": 2.8508686631943823e-05,
      "validation_loss": null
    },
    {
      "epoch": 1530,
      "train_loss": 2.7586229811681227e-05,
      "validation_loss": null
    },
    {
      "epoch": 1531,
      "train_loss": 2.481242093684462e-05,
      "validation_loss": null
    },
    {
      "epoch": 1532,
      "train_loss": 2.4317582857394074e-05,
      "validation_loss": null
    },
    {
      "epoch": 1533,
      "train_loss": 2.9405424356809817e-05,
      "validation_loss": null
    },
    {
      "epoch": 1534,
      "train_loss": 4.415329001656371e-05,
      "validation_loss": null
    },
    {
      "epoch": 1535,
      "train_loss": 4.0279588170556555e-05,
      "validation_loss": null
    },
    {
      "epoch": 1536,
      "train_loss": 3.6876425130887786e-05,
      "validation_loss": null
    },
    {
      "epoch": 1537,
      "train_loss": 3.845222029970804e-05,
      "validation_loss": null
    },
    {
      "epoch": 1538,
      "train_loss": 3.391352482948504e-05,
      "validation_loss": null
    },
    {
      "epoch": 1539,
      "train_loss": 3.288841859709161e-05,
      "validation_loss": null
    },
    {
      "epoch": 1540,
      "train_loss": 2.7083526219939813e-05,
      "validation_loss": null
    },
    {
      "epoch": 1541,
      "train_loss": 2.8201540165658418e-05,
      "validation_loss": null
    },
    {
      "epoch": 1542,
      "train_loss": 2.594088073237799e-05,
      "validation_loss": null
    },
    {
      "epoch": 1543,
      "train_loss": 2.7376081561669707e-05,
      "validation_loss": null
    },
    {
      "epoch": 1544,
      "train_loss": 3.9890283814101065e-05,
      "validation_loss": null
    },
    {
      "epoch": 1545,
      "train_loss": 3.7116912305161044e-05,
      "validation_loss": null
    },
    {
      "epoch": 1546,
      "train_loss": 3.132052148026155e-05,
      "validation_loss": null
    },
    {
      "epoch": 1547,
      "train_loss": 2.5954466764233075e-05,
      "validation_loss": null
    },
    {
      "epoch": 1548,
      "train_loss": 2.207045266307735e-05,
      "validation_loss": null
    },
    {
      "epoch": 1549,
      "train_loss": 2.2589873272712186e-05,
      "validation_loss": null
    },
    {
      "epoch": 1550,
      "train_loss": 2.3352198089317728e-05,
      "validation_loss": null
    },
    {
      "epoch": 1551,
      "train_loss": 2.647296747050455e-05,
      "validation_loss": null
    },
    {
      "epoch": 1552,
      "train_loss": 2.6412702734685605e-05,
      "validation_loss": null
    },
    {
      "epoch": 1553,
      "train_loss": 2.2358974698969785e-05,
      "validation_loss": null
    },
    {
      "epoch": 1554,
      "train_loss": 2.295008905801094e-05,
      "validation_loss": null
    },
    {
      "epoch": 1555,
      "train_loss": 1.9743308762877456e-05,
      "validation_loss": null
    },
    {
      "epoch": 1556,
      "train_loss": 1.8958749327187736e-05,
      "validation_loss": null
    },
    {
      "epoch": 1557,
      "train_loss": 1.9722451017716798e-05,
      "validation_loss": null
    },
    {
      "epoch": 1558,
      "train_loss": 2.325178623626319e-05,
      "validation_loss": null
    },
    {
      "epoch": 1559,
      "train_loss": 1.954293353871132e-05,
      "validation_loss": null
    },
    {
      "epoch": 1560,
      "train_loss": 1.952492018770297e-05,
      "validation_loss": null
    },
    {
      "epoch": 1561,
      "train_loss": 2.303304447397952e-05,
      "validation_loss": null
    },
    {
      "epoch": 1562,
      "train_loss": 2.2804586882961707e-05,
      "validation_loss": null
    },
    {
      "epoch": 1563,
      "train_loss": 2.378158145196115e-05,
      "validation_loss": null
    },
    {
      "epoch": 1564,
      "train_loss": 2.8526127304454956e-05,
      "validation_loss": null
    },
    {
      "epoch": 1565,
      "train_loss": 2.62084607432674e-05,
      "validation_loss": null
    },
    {
      "epoch": 1566,
      "train_loss": 2.5304423211815043e-05,
      "validation_loss": null
    },
    {
      "epoch": 1567,
      "train_loss": 3.092156637447058e-05,
      "validation_loss": null
    },
    {
      "epoch": 1568,
      "train_loss": 2.9418567767910037e-05,
      "validation_loss": null
    },
    {
      "epoch": 1569,
      "train_loss": 2.424843680475735e-05,
      "validation_loss": null
    },
    {
      "epoch": 1570,
      "train_loss": 2.4558739823987707e-05,
      "validation_loss": null
    },
    {
      "epoch": 1571,
      "train_loss": 2.285617099712706e-05,
      "validation_loss": null
    },
    {
      "epoch": 1572,
      "train_loss": 2.0984539939187623e-05,
      "validation_loss": null
    },
    {
      "epoch": 1573,
      "train_loss": 2.0865571463622877e-05,
      "validation_loss": null
    },
    {
      "epoch": 1574,
      "train_loss": 2.3330390629578484e-05,
      "validation_loss": null
    },
    {
      "epoch": 1575,
      "train_loss": 2.2516462195198983e-05,
      "validation_loss": null
    },
    {
      "epoch": 1576,
      "train_loss": 2.4155475330189802e-05,
      "validation_loss": null
    },
    {
      "epoch": 1577,
      "train_loss": 2.5191591097003158e-05,
      "validation_loss": null
    },
    {
      "epoch": 1578,
      "train_loss": 3.0172258953421584e-05,
      "validation_loss": null
    },
    {
      "epoch": 1579,
      "train_loss": 3.253359621642934e-05,
      "validation_loss": null
    },
    {
      "epoch": 1580,
      "train_loss": 2.785636449213295e-05,
      "validation_loss": null
    },
    {
      "epoch": 1581,
      "train_loss": 2.7715139973831054e-05,
      "validation_loss": null
    },
    {
      "epoch": 1582,
      "train_loss": 2.4104854067748722e-05,
      "validation_loss": null
    },
    {
      "epoch": 1583,
      "train_loss": 3.5512830638456056e-05,
      "validation_loss": null
    },
    {
      "epoch": 1584,
      "train_loss": 3.068239938228443e-05,
      "validation_loss": null
    },
    {
      "epoch": 1585,
      "train_loss": 2.8628295897053453e-05,
      "validation_loss": null
    },
    {
      "epoch": 1586,
      "train_loss": 3.641113815117731e-05,
      "validation_loss": null
    },
    {
      "epoch": 1587,
      "train_loss": 3.730943798372108e-05,
      "validation_loss": null
    },
    {
      "epoch": 1588,
      "train_loss": 3.654812381783914e-05,
      "validation_loss": null
    },
    {
      "epoch": 1589,
      "train_loss": 4.0449812836919184e-05,
      "validation_loss": null
    },
    {
      "epoch": 1590,
      "train_loss": 3.795965555279205e-05,
      "validation_loss": null
    },
    {
      "epoch": 1591,
      "train_loss": 3.952601360247677e-05,
      "validation_loss": null
    },
    {
      "epoch": 1592,
      "train_loss": 9.194561062031426e-05,
      "validation_loss": null
    },
    {
      "epoch": 1593,
      "train_loss": 7.908406769274734e-05,
      "validation_loss": null
    },
    {
      "epoch": 1594,
      "train_loss": 7.494230764374758e-05,
      "validation_loss": null
    },
    {
      "epoch": 1595,
      "train_loss": 5.7603169504242636e-05,
      "validation_loss": null
    },
    {
      "epoch": 1596,
      "train_loss": 5.858133044966962e-05,
      "validation_loss": null
    },
    {
      "epoch": 1597,
      "train_loss": 8.698253683784667e-05,
      "validation_loss": null
    },
    {
      "epoch": 1598,
      "train_loss": 7.54582635838435e-05,
      "validation_loss": null
    },
    {
      "epoch": 1599,
      "train_loss": 3.886289232468698e-05,
      "validation_loss": null
    },
    {
      "epoch": 1600,
      "train_loss": 3.4710254516944085e-05,
      "validation_loss": null
    },
    {
      "epoch": 1601,
      "train_loss": 3.17263390267423e-05,
      "validation_loss": null
    },
    {
      "epoch": 1602,
      "train_loss": 4.51722717116354e-05,
      "validation_loss": null
    },
    {
      "epoch": 1603,
      "train_loss": 5.344017780670482e-05,
      "validation_loss": null
    },
    {
      "epoch": 1604,
      "train_loss": 4.427654352588838e-05,
      "validation_loss": null
    },
    {
      "epoch": 1605,
      "train_loss": 3.4685516058945926e-05,
      "validation_loss": null
    },
    {
      "epoch": 1606,
      "train_loss": 5.0046684514705296e-05,
      "validation_loss": null
    },
    {
      "epoch": 1607,
      "train_loss": 3.377451584027666e-05,
      "validation_loss": null
    },
    {
      "epoch": 1608,
      "train_loss": 3.604539597290568e-05,
      "validation_loss": null
    },
    {
      "epoch": 1609,
      "train_loss": 4.7348429385844305e-05,
      "validation_loss": null
    },
    {
      "epoch": 1610,
      "train_loss": 3.272107905003294e-05,
      "validation_loss": null
    },
    {
      "epoch": 1611,
      "train_loss": 2.8776796372969533e-05,
      "validation_loss": null
    },
    {
      "epoch": 1612,
      "train_loss": 3.021676639036741e-05,
      "validation_loss": null
    },
    {
      "epoch": 1613,
      "train_loss": 2.4330548310697646e-05,
      "validation_loss": null
    },
    {
      "epoch": 1614,
      "train_loss": 5.64072142879013e-05,
      "validation_loss": null
    },
    {
      "epoch": 1615,
      "train_loss": 3.201559633857364e-05,
      "validation_loss": null
    },
    {
      "epoch": 1616,
      "train_loss": 5.131488675639654e-05,
      "validation_loss": null
    },
    {
      "epoch": 1617,
      "train_loss": 4.8972449601731365e-05,
      "validation_loss": null
    },
    {
      "epoch": 1618,
      "train_loss": 4.453049910403529e-05,
      "validation_loss": null
    },
    {
      "epoch": 1619,
      "train_loss": 4.801665879009912e-05,
      "validation_loss": null
    },
    {
      "epoch": 1620,
      "train_loss": 5.313490166574613e-05,
      "validation_loss": null
    },
    {
      "epoch": 1621,
      "train_loss": 6.39686044855302e-05,
      "validation_loss": null
    },
    {
      "epoch": 1622,
      "train_loss": 3.90902419490481e-05,
      "validation_loss": null
    },
    {
      "epoch": 1623,
      "train_loss": 3.937358613054837e-05,
      "validation_loss": null
    },
    {
      "epoch": 1624,
      "train_loss": 3.506573618021017e-05,
      "validation_loss": null
    },
    {
      "epoch": 1625,
      "train_loss": 2.5484031438180762e-05,
      "validation_loss": null
    },
    {
      "epoch": 1626,
      "train_loss": 5.1293647427681004e-05,
      "validation_loss": null
    },
    {
      "epoch": 1627,
      "train_loss": 6.062182001187466e-05,
      "validation_loss": null
    },
    {
      "epoch": 1628,
      "train_loss": 4.8779990417339526e-05,
      "validation_loss": null
    },
    {
      "epoch": 1629,
      "train_loss": 3.867082098925797e-05,
      "validation_loss": null
    },
    {
      "epoch": 1630,
      "train_loss": 6.437310387102318e-05,
      "validation_loss": null
    },
    {
      "epoch": 1631,
      "train_loss": 4.691381501389616e-05,
      "validation_loss": null
    },
    {
      "epoch": 1632,
      "train_loss": 6.495601802372321e-05,
      "validation_loss": null
    },
    {
      "epoch": 1633,
      "train_loss": 8.214285157413947e-05,
      "validation_loss": null
    },
    {
      "epoch": 1634,
      "train_loss": 7.037440284168244e-05,
      "validation_loss": null
    },
    {
      "epoch": 1635,
      "train_loss": 0.00013571248766513437,
      "validation_loss": null
    },
    {
      "epoch": 1636,
      "train_loss": 0.00010249282341747958,
      "validation_loss": null
    },
    {
      "epoch": 1637,
      "train_loss": 9.164490701449622e-05,
      "validation_loss": null
    },
    {
      "epoch": 1638,
      "train_loss": 5.2875288222114454e-05,
      "validation_loss": null
    },
    {
      "epoch": 1639,
      "train_loss": 5.105881710834284e-05,
      "validation_loss": null
    },
    {
      "epoch": 1640,
      "train_loss": 6.927905148283268e-05,
      "validation_loss": null
    },
    {
      "epoch": 1641,
      "train_loss": 0.00010412297746774534,
      "validation_loss": null
    },
    {
      "epoch": 1642,
      "train_loss": 0.00023407225752533931,
      "validation_loss": null
    },
    {
      "epoch": 1643,
      "train_loss": 0.0001570240064110193,
      "validation_loss": null
    },
    {
      "epoch": 1644,
      "train_loss": 0.00015170845047881207,
      "validation_loss": null
    },
    {
      "epoch": 1645,
      "train_loss": 0.00019642057223892253,
      "validation_loss": null
    },
    {
      "epoch": 1646,
      "train_loss": 0.00010923216662680109,
      "validation_loss": null
    },
    {
      "epoch": 1647,
      "train_loss": 8.283727462791528e-05,
      "validation_loss": null
    },
    {
      "epoch": 1648,
      "train_loss": 9.597595756834683e-05,
      "validation_loss": null
    },
    {
      "epoch": 1649,
      "train_loss": 6.812478669518087e-05,
      "validation_loss": null
    },
    {
      "epoch": 1650,
      "train_loss": 5.9717400694757496e-05,
      "validation_loss": null
    },
    {
      "epoch": 1651,
      "train_loss": 3.076810308913183e-05,
      "validation_loss": null
    },
    {
      "epoch": 1652,
      "train_loss": 4.370948348271971e-05,
      "validation_loss": null
    },
    {
      "epoch": 1653,
      "train_loss": 4.4243461969825956e-05,
      "validation_loss": null
    },
    {
      "epoch": 1654,
      "train_loss": 2.4911656712194803e-05,
      "validation_loss": null
    },
    {
      "epoch": 1655,
      "train_loss": 2.9282766263349913e-05,
      "validation_loss": null
    },
    {
      "epoch": 1656,
      "train_loss": 2.6333354981034063e-05,
      "validation_loss": null
    },
    {
      "epoch": 1657,
      "train_loss": 2.379459894857266e-05,
      "validation_loss": null
    },
    {
      "epoch": 1658,
      "train_loss": 2.234864647713645e-05,
      "validation_loss": null
    },
    {
      "epoch": 1659,
      "train_loss": 2.372515821965256e-05,
      "validation_loss": null
    },
    {
      "epoch": 1660,
      "train_loss": 2.201463717533948e-05,
      "validation_loss": null
    },
    {
      "epoch": 1661,
      "train_loss": 6.0256709629255864e-05,
      "validation_loss": null
    },
    {
      "epoch": 1662,
      "train_loss": 0.00013150123898716023,
      "validation_loss": null
    },
    {
      "epoch": 1663,
      "train_loss": 0.0001033076777174655,
      "validation_loss": null
    },
    {
      "epoch": 1664,
      "train_loss": 0.0001376091434697931,
      "validation_loss": null
    },
    {
      "epoch": 1665,
      "train_loss": 0.0001607082271625081,
      "validation_loss": null
    },
    {
      "epoch": 1666,
      "train_loss": 0.00024124171533104446,
      "validation_loss": null
    },
    {
      "epoch": 1667,
      "train_loss": 0.0003374031140184444,
      "validation_loss": null
    },
    {
      "epoch": 1668,
      "train_loss": 0.00037615769542753696,
      "validation_loss": null
    },
    {
      "epoch": 1669,
      "train_loss": 0.00024741169666716206,
      "validation_loss": null
    },
    {
      "epoch": 1670,
      "train_loss": 0.00024018149916729372,
      "validation_loss": null
    },
    {
      "epoch": 1671,
      "train_loss": 0.00013353205900380595,
      "validation_loss": null
    },
    {
      "epoch": 1672,
      "train_loss": 0.0001064684539515939,
      "validation_loss": null
    },
    {
      "epoch": 1673,
      "train_loss": 6.652970195217576e-05,
      "validation_loss": null
    },
    {
      "epoch": 1674,
      "train_loss": 3.797354313266826e-05,
      "validation_loss": null
    },
    {
      "epoch": 1675,
      "train_loss": 4.07753796025645e-05,
      "validation_loss": null
    },
    {
      "epoch": 1676,
      "train_loss": 6.557952312707332e-05,
      "validation_loss": null
    },
    {
      "epoch": 1677,
      "train_loss": 0.00014214592415050397,
      "validation_loss": null
    },
    {
      "epoch": 1678,
      "train_loss": 0.00018107854541287653,
      "validation_loss": null
    },
    {
      "epoch": 1679,
      "train_loss": 0.00010840674965745873,
      "validation_loss": null
    },
    {
      "epoch": 1680,
      "train_loss": 0.00018275916469671452,
      "validation_loss": null
    },
    {
      "epoch": 1681,
      "train_loss": 0.0001408028567513813,
      "validation_loss": null
    },
    {
      "epoch": 1682,
      "train_loss": 0.00021573667618213221,
      "validation_loss": null
    },
    {
      "epoch": 1683,
      "train_loss": 0.00020727187721705477,
      "validation_loss": null
    },
    {
      "epoch": 1684,
      "train_loss": 0.0002118972875501236,
      "validation_loss": null
    },
    {
      "epoch": 1685,
      "train_loss": 0.00018480478967022564,
      "validation_loss": null
    },
    {
      "epoch": 1686,
      "train_loss": 0.00015739536482012935,
      "validation_loss": null
    },
    {
      "epoch": 1687,
      "train_loss": 8.979470981608352e-05,
      "validation_loss": null
    },
    {
      "epoch": 1688,
      "train_loss": 6.230368590978388e-05,
      "validation_loss": null
    },
    {
      "epoch": 1689,
      "train_loss": 5.253641797430141e-05,
      "validation_loss": null
    },
    {
      "epoch": 1690,
      "train_loss": 9.323156336904503e-05,
      "validation_loss": null
    },
    {
      "epoch": 1691,
      "train_loss": 0.00011147871919092722,
      "validation_loss": null
    },
    {
      "epoch": 1692,
      "train_loss": 9.989760292228311e-05,
      "validation_loss": null
    },
    {
      "epoch": 1693,
      "train_loss": 9.105586367594596e-05,
      "validation_loss": null
    },
    {
      "epoch": 1694,
      "train_loss": 0.0001689843896504802,
      "validation_loss": null
    },
    {
      "epoch": 1695,
      "train_loss": 0.00018070563399103575,
      "validation_loss": null
    },
    {
      "epoch": 1696,
      "train_loss": 0.0003935153072234243,
      "validation_loss": null
    },
    {
      "epoch": 1697,
      "train_loss": 0.00038416058911631507,
      "validation_loss": null
    },
    {
      "epoch": 1698,
      "train_loss": 0.00032316341013130214,
      "validation_loss": null
    },
    {
      "epoch": 1699,
      "train_loss": 0.0002638314633966527,
      "validation_loss": null
    },
    {
      "epoch": 1700,
      "train_loss": 0.0003422334456344187,
      "validation_loss": null
    },
    {
      "epoch": 1701,
      "train_loss": 0.0004134001970265268,
      "validation_loss": null
    },
    {
      "epoch": 1702,
      "train_loss": 0.00031899846605180454,
      "validation_loss": null
    },
    {
      "epoch": 1703,
      "train_loss": 0.00010156153883953165,
      "validation_loss": null
    },
    {
      "epoch": 1704,
      "train_loss": 5.7983741296791574e-05,
      "validation_loss": null
    },
    {
      "epoch": 1705,
      "train_loss": 4.425416126549761e-05,
      "validation_loss": null
    },
    {
      "epoch": 1706,
      "train_loss": 3.76809022984364e-05,
      "validation_loss": null
    },
    {
      "epoch": 1707,
      "train_loss": 3.854305113135423e-05,
      "validation_loss": null
    },
    {
      "epoch": 1708,
      "train_loss": 6.149950503539812e-05,
      "validation_loss": null
    },
    {
      "epoch": 1709,
      "train_loss": 4.275844023343072e-05,
      "validation_loss": null
    },
    {
      "epoch": 1710,
      "train_loss": 5.0702619773801416e-05,
      "validation_loss": null
    },
    {
      "epoch": 1711,
      "train_loss": 9.15746176259644e-05,
      "validation_loss": null
    },
    {
      "epoch": 1712,
      "train_loss": 6.733244986713139e-05,
      "validation_loss": null
    },
    {
      "epoch": 1713,
      "train_loss": 9.265221140392694e-05,
      "validation_loss": null
    },
    {
      "epoch": 1714,
      "train_loss": 8.598192006401304e-05,
      "validation_loss": null
    },
    {
      "epoch": 1715,
      "train_loss": 5.257171546367722e-05,
      "validation_loss": null
    },
    {
      "epoch": 1716,
      "train_loss": 4.48557907980608e-05,
      "validation_loss": null
    },
    {
      "epoch": 1717,
      "train_loss": 3.355796838554347e-05,
      "validation_loss": null
    },
    {
      "epoch": 1718,
      "train_loss": 2.5839103424611192e-05,
      "validation_loss": null
    },
    {
      "epoch": 1719,
      "train_loss": 2.0620358352769268e-05,
      "validation_loss": null
    },
    {
      "epoch": 1720,
      "train_loss": 2.140531372181916e-05,
      "validation_loss": null
    },
    {
      "epoch": 1721,
      "train_loss": 2.339906910088353e-05,
      "validation_loss": null
    },
    {
      "epoch": 1722,
      "train_loss": 1.9631040939000537e-05,
      "validation_loss": null
    },
    {
      "epoch": 1723,
      "train_loss": 1.9896115291986563e-05,
      "validation_loss": null
    },
    {
      "epoch": 1724,
      "train_loss": 2.1895962264453476e-05,
      "validation_loss": null
    },
    {
      "epoch": 1725,
      "train_loss": 1.8791589253750215e-05,
      "validation_loss": null
    },
    {
      "epoch": 1726,
      "train_loss": 2.117618724797568e-05,
      "validation_loss": null
    },
    {
      "epoch": 1727,
      "train_loss": 1.7915569919245576e-05,
      "validation_loss": null
    },
    {
      "epoch": 1728,
      "train_loss": 1.9568723170474794e-05,
      "validation_loss": null
    },
    {
      "epoch": 1729,
      "train_loss": 1.5585079078543156e-05,
      "validation_loss": null
    },
    {
      "epoch": 1730,
      "train_loss": 2.089331751954483e-05,
      "validation_loss": null
    },
    {
      "epoch": 1731,
      "train_loss": 3.319107498908933e-05,
      "validation_loss": null
    },
    {
      "epoch": 1732,
      "train_loss": 3.63891415165401e-05,
      "validation_loss": null
    },
    {
      "epoch": 1733,
      "train_loss": 2.7268928331775693e-05,
      "validation_loss": null
    },
    {
      "epoch": 1734,
      "train_loss": 1.9987760020335878e-05,
      "validation_loss": null
    },
    {
      "epoch": 1735,
      "train_loss": 2.8345473563402062e-05,
      "validation_loss": null
    },
    {
      "epoch": 1736,
      "train_loss": 4.6149762258412214e-05,
      "validation_loss": null
    },
    {
      "epoch": 1737,
      "train_loss": 6.0931333185484014e-05,
      "validation_loss": null
    },
    {
      "epoch": 1738,
      "train_loss": 7.837614834392702e-05,
      "validation_loss": null
    },
    {
      "epoch": 1739,
      "train_loss": 6.663172098342329e-05,
      "validation_loss": null
    },
    {
      "epoch": 1740,
      "train_loss": 4.496983860412406e-05,
      "validation_loss": null
    },
    {
      "epoch": 1741,
      "train_loss": 5.6757820857455954e-05,
      "validation_loss": null
    },
    {
      "epoch": 1742,
      "train_loss": 5.0181650092579526e-05,
      "validation_loss": null
    },
    {
      "epoch": 1743,
      "train_loss": 4.7855411644882726e-05,
      "validation_loss": null
    },
    {
      "epoch": 1744,
      "train_loss": 0.00011905963482503366,
      "validation_loss": null
    },
    {
      "epoch": 1745,
      "train_loss": 0.00011507534347603926,
      "validation_loss": null
    },
    {
      "epoch": 1746,
      "train_loss": 0.0002804010623044127,
      "validation_loss": null
    },
    {
      "epoch": 1747,
      "train_loss": 0.0004125718130833573,
      "validation_loss": null
    },
    {
      "epoch": 1748,
      "train_loss": 0.0002143608128082835,
      "validation_loss": null
    },
    {
      "epoch": 1749,
      "train_loss": 0.00031115238056776836,
      "validation_loss": null
    },
    {
      "epoch": 1750,
      "train_loss": 0.00036320470583935577,
      "validation_loss": null
    },
    {
      "epoch": 1751,
      "train_loss": 0.0003102550593515237,
      "validation_loss": null
    },
    {
      "epoch": 1752,
      "train_loss": 0.00026411054002690234,
      "validation_loss": null
    },
    {
      "epoch": 1753,
      "train_loss": 0.00024553932922167913,
      "validation_loss": null
    },
    {
      "epoch": 1754,
      "train_loss": 0.00016051655499419817,
      "validation_loss": null
    },
    {
      "epoch": 1755,
      "train_loss": 0.00013723757001571357,
      "validation_loss": null
    },
    {
      "epoch": 1756,
      "train_loss": 0.00012176531405809025,
      "validation_loss": null
    },
    {
      "epoch": 1757,
      "train_loss": 0.000155082775260477,
      "validation_loss": null
    },
    {
      "epoch": 1758,
      "train_loss": 0.00012686530923626074,
      "validation_loss": null
    },
    {
      "epoch": 1759,
      "train_loss": 0.00013466822484689247,
      "validation_loss": null
    },
    {
      "epoch": 1760,
      "train_loss": 0.0001874763563844479,
      "validation_loss": null
    },
    {
      "epoch": 1761,
      "train_loss": 0.0001759796586055826,
      "validation_loss": null
    },
    {
      "epoch": 1762,
      "train_loss": 0.00016830579301717484,
      "validation_loss": null
    },
    {
      "epoch": 1763,
      "train_loss": 0.00023378229404139952,
      "validation_loss": null
    },
    {
      "epoch": 1764,
      "train_loss": 0.0001758528346221687,
      "validation_loss": null
    },
    {
      "epoch": 1765,
      "train_loss": 0.0001223989335509638,
      "validation_loss": null
    },
    {
      "epoch": 1766,
      "train_loss": 0.00012293349290201958,
      "validation_loss": null
    },
    {
      "epoch": 1767,
      "train_loss": 0.00011011422611773014,
      "validation_loss": null
    },
    {
      "epoch": 1768,
      "train_loss": 5.659524079722663e-05,
      "validation_loss": null
    },
    {
      "epoch": 1769,
      "train_loss": 8.068315017024158e-05,
      "validation_loss": null
    },
    {
      "epoch": 1770,
      "train_loss": 0.00013002400131275257,
      "validation_loss": null
    },
    {
      "epoch": 1771,
      "train_loss": 0.00010658346986423971,
      "validation_loss": null
    },
    {
      "epoch": 1772,
      "train_loss": 0.00010809700809962426,
      "validation_loss": null
    },
    {
      "epoch": 1773,
      "train_loss": 0.00010779921528107177,
      "validation_loss": null
    },
    {
      "epoch": 1774,
      "train_loss": 0.00011090797569421638,
      "validation_loss": null
    },
    {
      "epoch": 1775,
      "train_loss": 7.749035042555381e-05,
      "validation_loss": null
    },
    {
      "epoch": 1776,
      "train_loss": 4.729795424888531e-05,
      "validation_loss": null
    },
    {
      "epoch": 1777,
      "train_loss": 4.508761331736524e-05,
      "validation_loss": null
    },
    {
      "epoch": 1778,
      "train_loss": 4.240602922638775e-05,
      "validation_loss": null
    },
    {
      "epoch": 1779,
      "train_loss": 8.048603194765747e-05,
      "validation_loss": null
    },
    {
      "epoch": 1780,
      "train_loss": 7.474097017418696e-05,
      "validation_loss": null
    },
    {
      "epoch": 1781,
      "train_loss": 0.00012703638357278478,
      "validation_loss": null
    },
    {
      "epoch": 1782,
      "train_loss": 9.053389173156272e-05,
      "validation_loss": null
    },
    {
      "epoch": 1783,
      "train_loss": 8.793837575164314e-05,
      "validation_loss": null
    },
    {
      "epoch": 1784,
      "train_loss": 9.819328099385732e-05,
      "validation_loss": null
    },
    {
      "epoch": 1785,
      "train_loss": 0.00014673166394156092,
      "validation_loss": null
    },
    {
      "epoch": 1786,
      "train_loss": 0.00012126368690385587,
      "validation_loss": null
    },
    {
      "epoch": 1787,
      "train_loss": 0.0002103451382329998,
      "validation_loss": null
    },
    {
      "epoch": 1788,
      "train_loss": 0.0001738829887472093,
      "validation_loss": null
    },
    {
      "epoch": 1789,
      "train_loss": 0.0001321834780456912,
      "validation_loss": null
    },
    {
      "epoch": 1790,
      "train_loss": 9.621496590423501e-05,
      "validation_loss": null
    },
    {
      "epoch": 1791,
      "train_loss": 0.0001655274447532267,
      "validation_loss": null
    },
    {
      "epoch": 1792,
      "train_loss": 0.0001496579174676703,
      "validation_loss": null
    },
    {
      "epoch": 1793,
      "train_loss": 0.00014269138207762607,
      "validation_loss": null
    },
    {
      "epoch": 1794,
      "train_loss": 0.00011224300720237402,
      "validation_loss": null
    },
    {
      "epoch": 1795,
      "train_loss": 0.00012610672153338479,
      "validation_loss": null
    },
    {
      "epoch": 1796,
      "train_loss": 0.00016187821246502508,
      "validation_loss": null
    },
    {
      "epoch": 1797,
      "train_loss": 9.02708893085623e-05,
      "validation_loss": null
    },
    {
      "epoch": 1798,
      "train_loss": 7.547384908927295e-05,
      "validation_loss": null
    },
    {
      "epoch": 1799,
      "train_loss": 5.1632927352329716e-05,
      "validation_loss": null
    },
    {
      "epoch": 1800,
      "train_loss": 6.342455910473493e-05,
      "validation_loss": null
    },
    {
      "epoch": 1801,
      "train_loss": 4.536714914138429e-05,
      "validation_loss": null
    },
    {
      "epoch": 1802,
      "train_loss": 4.456872658112035e-05,
      "validation_loss": null
    },
    {
      "epoch": 1803,
      "train_loss": 4.557674967347541e-05,
      "validation_loss": null
    },
    {
      "epoch": 1804,
      "train_loss": 5.980526192514743e-05,
      "validation_loss": null
    },
    {
      "epoch": 1805,
      "train_loss": 0.00014694053526101116,
      "validation_loss": null
    },
    {
      "epoch": 1806,
      "train_loss": 0.00011964629326636593,
      "validation_loss": null
    },
    {
      "epoch": 1807,
      "train_loss": 0.00010197068210497189,
      "validation_loss": null
    },
    {
      "epoch": 1808,
      "train_loss": 0.00010440135237230506,
      "validation_loss": null
    },
    {
      "epoch": 1809,
      "train_loss": 8.006413392851957e-05,
      "validation_loss": null
    },
    {
      "epoch": 1810,
      "train_loss": 6.608304789147546e-05,
      "validation_loss": null
    },
    {
      "epoch": 1811,
      "train_loss": 7.883718232430208e-05,
      "validation_loss": null
    },
    {
      "epoch": 1812,
      "train_loss": 8.404936428026606e-05,
      "validation_loss": null
    },
    {
      "epoch": 1813,
      "train_loss": 0.00011211269399306427,
      "validation_loss": null
    },
    {
      "epoch": 1814,
      "train_loss": 9.412641803565848e-05,
      "validation_loss": null
    },
    {
      "epoch": 1815,
      "train_loss": 8.464249832387496e-05,
      "validation_loss": null
    },
    {
      "epoch": 1816,
      "train_loss": 9.461647762994592e-05,
      "validation_loss": null
    },
    {
      "epoch": 1817,
      "train_loss": 8.796948612952192e-05,
      "validation_loss": null
    },
    {
      "epoch": 1818,
      "train_loss": 0.0001233032679819088,
      "validation_loss": null
    },
    {
      "epoch": 1819,
      "train_loss": 9.036607176919158e-05,
      "validation_loss": null
    },
    {
      "epoch": 1820,
      "train_loss": 6.895207964892809e-05,
      "validation_loss": null
    },
    {
      "epoch": 1821,
      "train_loss": 0.00013330616800683655,
      "validation_loss": null
    },
    {
      "epoch": 1822,
      "train_loss": 0.00018653989819641638,
      "validation_loss": null
    },
    {
      "epoch": 1823,
      "train_loss": 0.00026782398506636836,
      "validation_loss": null
    },
    {
      "epoch": 1824,
      "train_loss": 0.00021822748789822476,
      "validation_loss": null
    },
    {
      "epoch": 1825,
      "train_loss": 0.0002735474384583843,
      "validation_loss": null
    },
    {
      "epoch": 1826,
      "train_loss": 0.00035169721458158974,
      "validation_loss": null
    },
    {
      "epoch": 1827,
      "train_loss": 0.0003350699488590989,
      "validation_loss": null
    },
    {
      "epoch": 1828,
      "train_loss": 0.00022982800227408815,
      "validation_loss": null
    },
    {
      "epoch": 1829,
      "train_loss": 0.00017315357399638742,
      "validation_loss": null
    },
    {
      "epoch": 1830,
      "train_loss": 0.000166098708480907,
      "validation_loss": null
    },
    {
      "epoch": 1831,
      "train_loss": 0.000243444833257753,
      "validation_loss": null
    },
    {
      "epoch": 1832,
      "train_loss": 0.00015572690851210305,
      "validation_loss": null
    },
    {
      "epoch": 1833,
      "train_loss": 8.554915216841942e-05,
      "validation_loss": null
    },
    {
      "epoch": 1834,
      "train_loss": 6.692343312250968e-05,
      "validation_loss": null
    },
    {
      "epoch": 1835,
      "train_loss": 0.00010061920329462737,
      "validation_loss": null
    },
    {
      "epoch": 1836,
      "train_loss": 0.00016653575666067708,
      "validation_loss": null
    },
    {
      "epoch": 1837,
      "train_loss": 0.0001765353048944639,
      "validation_loss": null
    },
    {
      "epoch": 1838,
      "train_loss": 0.0001427128324091124,
      "validation_loss": null
    },
    {
      "epoch": 1839,
      "train_loss": 0.0001582554638540993,
      "validation_loss": null
    },
    {
      "epoch": 1840,
      "train_loss": 0.0002185257868000513,
      "validation_loss": null
    },
    {
      "epoch": 1841,
      "train_loss": 0.00019071548498080423,
      "validation_loss": null
    },
    {
      "epoch": 1842,
      "train_loss": 0.00027643077515272633,
      "validation_loss": null
    },
    {
      "epoch": 1843,
      "train_loss": 0.0004392110551836797,
      "validation_loss": null
    },
    {
      "epoch": 1844,
      "train_loss": 0.0004751019684287409,
      "validation_loss": null
    },
    {
      "epoch": 1845,
      "train_loss": 0.0005053806768652672,
      "validation_loss": null
    },
    {
      "epoch": 1846,
      "train_loss": 0.00038558027396599454,
      "validation_loss": null
    },
    {
      "epoch": 1847,
      "train_loss": 0.0002606797610901089,
      "validation_loss": null
    },
    {
      "epoch": 1848,
      "train_loss": 0.0001883989284073727,
      "validation_loss": null
    },
    {
      "epoch": 1849,
      "train_loss": 0.00029119529709633853,
      "validation_loss": null
    },
    {
      "epoch": 1850,
      "train_loss": 0.000397073265857115,
      "validation_loss": null
    },
    {
      "epoch": 1851,
      "train_loss": 0.00037512093452581513,
      "validation_loss": null
    },
    {
      "epoch": 1852,
      "train_loss": 0.00032009423496977735,
      "validation_loss": null
    },
    {
      "epoch": 1853,
      "train_loss": 0.0003462368105020788,
      "validation_loss": null
    },
    {
      "epoch": 1854,
      "train_loss": 0.0009988766396418214,
      "validation_loss": null
    },
    {
      "epoch": 1855,
      "train_loss": 0.0006464083627280262,
      "validation_loss": null
    },
    {
      "epoch": 1856,
      "train_loss": 0.0006036859364636863,
      "validation_loss": null
    },
    {
      "epoch": 1857,
      "train_loss": 0.0004914892908547901,
      "validation_loss": null
    },
    {
      "epoch": 1858,
      "train_loss": 0.0003788063541934308,
      "validation_loss": null
    },
    {
      "epoch": 1859,
      "train_loss": 0.00032848275441210717,
      "validation_loss": null
    },
    {
      "epoch": 1860,
      "train_loss": 0.0002564749415291266,
      "validation_loss": null
    },
    {
      "epoch": 1861,
      "train_loss": 0.00021170268155401573,
      "validation_loss": null
    },
    {
      "epoch": 1862,
      "train_loss": 0.0001676041938480921,
      "validation_loss": null
    },
    {
      "epoch": 1863,
      "train_loss": 0.0001772820032783784,
      "validation_loss": null
    },
    {
      "epoch": 1864,
      "train_loss": 0.00013992944231075753,
      "validation_loss": null
    },
    {
      "epoch": 1865,
      "train_loss": 0.000117800180305494,
      "validation_loss": null
    },
    {
      "epoch": 1866,
      "train_loss": 0.00013533315763601827,
      "validation_loss": null
    },
    {
      "epoch": 1867,
      "train_loss": 0.00010879562776406399,
      "validation_loss": null
    },
    {
      "epoch": 1868,
      "train_loss": 0.00010012548389366,
      "validation_loss": null
    },
    {
      "epoch": 1869,
      "train_loss": 0.00014487554451787018,
      "validation_loss": null
    },
    {
      "epoch": 1870,
      "train_loss": 0.00012182910561225274,
      "validation_loss": null
    },
    {
      "epoch": 1871,
      "train_loss": 0.00015484356441043524,
      "validation_loss": null
    },
    {
      "epoch": 1872,
      "train_loss": 0.00014103998142268715,
      "validation_loss": null
    },
    {
      "epoch": 1873,
      "train_loss": 6.540947571112256e-05,
      "validation_loss": null
    },
    {
      "epoch": 1874,
      "train_loss": 6.0541999927307996e-05,
      "validation_loss": null
    },
    {
      "epoch": 1875,
      "train_loss": 6.51968613156997e-05,
      "validation_loss": null
    },
    {
      "epoch": 1876,
      "train_loss": 5.03625506098615e-05,
      "validation_loss": null
    },
    {
      "epoch": 1877,
      "train_loss": 4.163088912415939e-05,
      "validation_loss": null
    },
    {
      "epoch": 1878,
      "train_loss": 4.3513279403365836e-05,
      "validation_loss": null
    },
    {
      "epoch": 1879,
      "train_loss": 3.0344619315453907e-05,
      "validation_loss": null
    },
    {
      "epoch": 1880,
      "train_loss": 2.180133065444857e-05,
      "validation_loss": null
    },
    {
      "epoch": 1881,
      "train_loss": 2.486082929762132e-05,
      "validation_loss": null
    },
    {
      "epoch": 1882,
      "train_loss": 2.0157471832741674e-05,
      "validation_loss": null
    },
    {
      "epoch": 1883,
      "train_loss": 1.9035505829378963e-05,
      "validation_loss": null
    },
    {
      "epoch": 1884,
      "train_loss": 1.3708503805294944e-05,
      "validation_loss": null
    },
    {
      "epoch": 1885,
      "train_loss": 1.4034885073973177e-05,
      "validation_loss": null
    },
    {
      "epoch": 1886,
      "train_loss": 1.2665522566142803e-05,
      "validation_loss": null
    },
    {
      "epoch": 1887,
      "train_loss": 1.4036729731338306e-05,
      "validation_loss": null
    },
    {
      "epoch": 1888,
      "train_loss": 1.4622565888405208e-05,
      "validation_loss": null
    },
    {
      "epoch": 1889,
      "train_loss": 1.5873700350413048e-05,
      "validation_loss": null
    },
    {
      "epoch": 1890,
      "train_loss": 1.4509222637570929e-05,
      "validation_loss": null
    },
    {
      "epoch": 1891,
      "train_loss": 2.0471829985682336e-05,
      "validation_loss": null
    },
    {
      "epoch": 1892,
      "train_loss": 1.4281798131479365e-05,
      "validation_loss": null
    },
    {
      "epoch": 1893,
      "train_loss": 1.7109450911650332e-05,
      "validation_loss": null
    },
    {
      "epoch": 1894,
      "train_loss": 2.5583668603859325e-05,
      "validation_loss": null
    },
    {
      "epoch": 1895,
      "train_loss": 3.272534720761339e-05,
      "validation_loss": null
    },
    {
      "epoch": 1896,
      "train_loss": 3.876369575866395e-05,
      "validation_loss": null
    },
    {
      "epoch": 1897,
      "train_loss": 3.3213100121227195e-05,
      "validation_loss": null
    },
    {
      "epoch": 1898,
      "train_loss": 4.128845001913659e-05,
      "validation_loss": null
    },
    {
      "epoch": 1899,
      "train_loss": 3.7296554485995635e-05,
      "validation_loss": null
    },
    {
      "epoch": 1900,
      "train_loss": 4.148327853828151e-05,
      "validation_loss": null
    },
    {
      "epoch": 1901,
      "train_loss": 4.948875397935303e-05,
      "validation_loss": null
    },
    {
      "epoch": 1902,
      "train_loss": 3.51439278650408e-05,
      "validation_loss": null
    },
    {
      "epoch": 1903,
      "train_loss": 3.6749562544476554e-05,
      "validation_loss": null
    },
    {
      "epoch": 1904,
      "train_loss": 3.0993826334856247e-05,
      "validation_loss": null
    },
    {
      "epoch": 1905,
      "train_loss": 4.0822193518074025e-05,
      "validation_loss": null
    },
    {
      "epoch": 1906,
      "train_loss": 4.16646657362839e-05,
      "validation_loss": null
    },
    {
      "epoch": 1907,
      "train_loss": 3.5650643136856765e-05,
      "validation_loss": null
    },
    {
      "epoch": 1908,
      "train_loss": 3.4899314616470496e-05,
      "validation_loss": null
    },
    {
      "epoch": 1909,
      "train_loss": 4.724915035896831e-05,
      "validation_loss": null
    },
    {
      "epoch": 1910,
      "train_loss": 3.4403713572020126e-05,
      "validation_loss": null
    },
    {
      "epoch": 1911,
      "train_loss": 3.1563273498452167e-05,
      "validation_loss": null
    },
    {
      "epoch": 1912,
      "train_loss": 3.624084112945841e-05,
      "validation_loss": null
    },
    {
      "epoch": 1913,
      "train_loss": 7.950886033035608e-05,
      "validation_loss": null
    },
    {
      "epoch": 1914,
      "train_loss": 8.89293862403267e-05,
      "validation_loss": null
    },
    {
      "epoch": 1915,
      "train_loss": 0.0001336206195345666,
      "validation_loss": null
    },
    {
      "epoch": 1916,
      "train_loss": 0.00013385017761417353,
      "validation_loss": null
    },
    {
      "epoch": 1917,
      "train_loss": 9.387403648967545e-05,
      "validation_loss": null
    },
    {
      "epoch": 1918,
      "train_loss": 7.693728710162557e-05,
      "validation_loss": null
    },
    {
      "epoch": 1919,
      "train_loss": 5.5630641630462684e-05,
      "validation_loss": null
    },
    {
      "epoch": 1920,
      "train_loss": 9.419987367841208e-05,
      "validation_loss": null
    },
    {
      "epoch": 1921,
      "train_loss": 0.00012345454928840304,
      "validation_loss": null
    },
    {
      "epoch": 1922,
      "train_loss": 7.868265634493178e-05,
      "validation_loss": null
    },
    {
      "epoch": 1923,
      "train_loss": 8.875318922542242e-05,
      "validation_loss": null
    },
    {
      "epoch": 1924,
      "train_loss": 8.375326595139793e-05,
      "validation_loss": null
    },
    {
      "epoch": 1925,
      "train_loss": 6.587062267095057e-05,
      "validation_loss": null
    },
    {
      "epoch": 1926,
      "train_loss": 7.407772176924886e-05,
      "validation_loss": null
    },
    {
      "epoch": 1927,
      "train_loss": 6.16501311014872e-05,
      "validation_loss": null
    },
    {
      "epoch": 1928,
      "train_loss": 0.00010078699091940912,
      "validation_loss": null
    },
    {
      "epoch": 1929,
      "train_loss": 0.00023541925313313387,
      "validation_loss": null
    },
    {
      "epoch": 1930,
      "train_loss": 0.0005988800678298705,
      "validation_loss": null
    },
    {
      "epoch": 1931,
      "train_loss": 0.00041671468352433294,
      "validation_loss": null
    },
    {
      "epoch": 1932,
      "train_loss": 0.0004882627852364547,
      "validation_loss": null
    },
    {
      "epoch": 1933,
      "train_loss": 0.0006297356352054825,
      "validation_loss": null
    },
    {
      "epoch": 1934,
      "train_loss": 0.0006954751475455447,
      "validation_loss": null
    },
    {
      "epoch": 1935,
      "train_loss": 0.0003375470971352317,
      "validation_loss": null
    },
    {
      "epoch": 1936,
      "train_loss": 0.00023902341005547592,
      "validation_loss": null
    },
    {
      "epoch": 1937,
      "train_loss": 0.0003118638237032832,
      "validation_loss": null
    },
    {
      "epoch": 1938,
      "train_loss": 0.00029734985663607303,
      "validation_loss": null
    },
    {
      "epoch": 1939,
      "train_loss": 0.00023083881630251804,
      "validation_loss": null
    },
    {
      "epoch": 1940,
      "train_loss": 0.00023957572718952887,
      "validation_loss": null
    },
    {
      "epoch": 1941,
      "train_loss": 0.00016616362295786125,
      "validation_loss": null
    },
    {
      "epoch": 1942,
      "train_loss": 0.0003029383668844174,
      "validation_loss": null
    },
    {
      "epoch": 1943,
      "train_loss": 0.0006676350862512158,
      "validation_loss": null
    },
    {
      "epoch": 1944,
      "train_loss": 0.0005468430010498398,
      "validation_loss": null
    },
    {
      "epoch": 1945,
      "train_loss": 0.00041758962874559476,
      "validation_loss": null
    },
    {
      "epoch": 1946,
      "train_loss": 0.0005719640240487126,
      "validation_loss": null
    },
    {
      "epoch": 1947,
      "train_loss": 0.00031810032022703026,
      "validation_loss": null
    },
    {
      "epoch": 1948,
      "train_loss": 0.0002735299963711037,
      "validation_loss": null
    },
    {
      "epoch": 1949,
      "train_loss": 0.0003726649350331475,
      "validation_loss": null
    },
    {
      "epoch": 1950,
      "train_loss": 0.0002658094292403095,
      "validation_loss": null
    },
    {
      "epoch": 1951,
      "train_loss": 0.00013803118539120382,
      "validation_loss": null
    },
    {
      "epoch": 1952,
      "train_loss": 9.28182319815581e-05,
      "validation_loss": null
    },
    {
      "epoch": 1953,
      "train_loss": 8.250003803469654e-05,
      "validation_loss": null
    },
    {
      "epoch": 1954,
      "train_loss": 6.727457892136752e-05,
      "validation_loss": null
    },
    {
      "epoch": 1955,
      "train_loss": 0.00010391620384729726,
      "validation_loss": null
    },
    {
      "epoch": 1956,
      "train_loss": 9.560300127810074e-05,
      "validation_loss": null
    },
    {
      "epoch": 1957,
      "train_loss": 0.00010087579600318956,
      "validation_loss": null
    },
    {
      "epoch": 1958,
      "train_loss": 0.00010585940193171457,
      "validation_loss": null
    },
    {
      "epoch": 1959,
      "train_loss": 5.391757844740318e-05,
      "validation_loss": null
    },
    {
      "epoch": 1960,
      "train_loss": 3.639134896123627e-05,
      "validation_loss": null
    },
    {
      "epoch": 1961,
      "train_loss": 3.970628717474433e-05,
      "validation_loss": null
    },
    {
      "epoch": 1962,
      "train_loss": 4.0847673113199155e-05,
      "validation_loss": null
    },
    {
      "epoch": 1963,
      "train_loss": 4.2534636627856846e-05,
      "validation_loss": null
    },
    {
      "epoch": 1964,
      "train_loss": 8.215765208458631e-05,
      "validation_loss": null
    },
    {
      "epoch": 1965,
      "train_loss": 8.673705411234146e-05,
      "validation_loss": null
    },
    {
      "epoch": 1966,
      "train_loss": 6.205737615042987e-05,
      "validation_loss": null
    },
    {
      "epoch": 1967,
      "train_loss": 5.676647439637842e-05,
      "validation_loss": null
    },
    {
      "epoch": 1968,
      "train_loss": 3.643529210724713e-05,
      "validation_loss": null
    },
    {
      "epoch": 1969,
      "train_loss": 6.925040886724471e-05,
      "validation_loss": null
    },
    {
      "epoch": 1970,
      "train_loss": 7.206717236840632e-05,
      "validation_loss": null
    },
    {
      "epoch": 1971,
      "train_loss": 6.754441932975574e-05,
      "validation_loss": null
    },
    {
      "epoch": 1972,
      "train_loss": 5.2475300738782e-05,
      "validation_loss": null
    },
    {
      "epoch": 1973,
      "train_loss": 3.422556963212426e-05,
      "validation_loss": null
    },
    {
      "epoch": 1974,
      "train_loss": 3.754179569013003e-05,
      "validation_loss": null
    },
    {
      "epoch": 1975,
      "train_loss": 5.336186465380403e-05,
      "validation_loss": null
    },
    {
      "epoch": 1976,
      "train_loss": 5.4101874209461836e-05,
      "validation_loss": null
    },
    {
      "epoch": 1977,
      "train_loss": 3.8332962705883096e-05,
      "validation_loss": null
    },
    {
      "epoch": 1978,
      "train_loss": 5.49187926582009e-05,
      "validation_loss": null
    },
    {
      "epoch": 1979,
      "train_loss": 6.421681569514071e-05,
      "validation_loss": null
    },
    {
      "epoch": 1980,
      "train_loss": 4.73148067814893e-05,
      "validation_loss": null
    },
    {
      "epoch": 1981,
      "train_loss": 3.925205318308953e-05,
      "validation_loss": null
    },
    {
      "epoch": 1982,
      "train_loss": 3.258162198310149e-05,
      "validation_loss": null
    },
    {
      "epoch": 1983,
      "train_loss": 4.111062197706714e-05,
      "validation_loss": null
    },
    {
      "epoch": 1984,
      "train_loss": 4.745857040688861e-05,
      "validation_loss": null
    },
    {
      "epoch": 1985,
      "train_loss": 0.00011630023234627313,
      "validation_loss": null
    },
    {
      "epoch": 1986,
      "train_loss": 0.00011147245499564128,
      "validation_loss": null
    },
    {
      "epoch": 1987,
      "train_loss": 8.677270266667215e-05,
      "validation_loss": null
    },
    {
      "epoch": 1988,
      "train_loss": 6.119054119658863e-05,
      "validation_loss": null
    },
    {
      "epoch": 1989,
      "train_loss": 5.248982375228984e-05,
      "validation_loss": null
    },
    {
      "epoch": 1990,
      "train_loss": 2.8200119737044184e-05,
      "validation_loss": null
    },
    {
      "epoch": 1991,
      "train_loss": 2.5610015048490215e-05,
      "validation_loss": null
    },
    {
      "epoch": 1992,
      "train_loss": 4.14635075382345e-05,
      "validation_loss": null
    },
    {
      "epoch": 1993,
      "train_loss": 4.169345225414468e-05,
      "validation_loss": null
    },
    {
      "epoch": 1994,
      "train_loss": 2.7513594937368503e-05,
      "validation_loss": null
    },
    {
      "epoch": 1995,
      "train_loss": 2.5255720680029804e-05,
      "validation_loss": null
    },
    {
      "epoch": 1996,
      "train_loss": 2.478022250013762e-05,
      "validation_loss": null
    },
    {
      "epoch": 1997,
      "train_loss": 1.6680541597856467e-05,
      "validation_loss": null
    },
    {
      "epoch": 1998,
      "train_loss": 1.5256507746623052e-05,
      "validation_loss": null
    },
    {
      "epoch": 1999,
      "train_loss": 1.5422112179900676e-05,
      "validation_loss": null
    },
    {
      "epoch": 2000,
      "train_loss": 1.7956006862126134e-05,
      "validation_loss": null
    }
  ],
  "target_statistics_train_only": {
    "mean": -25.585215168861424,
    "std": 49.7155473679296,
    "min": -155.53724670410156,
    "p1": -120.9492308807373,
    "p10": -120.0625213623047,
    "median": 3.0638864040374756,
    "p90": 10.19877576828003,
    "p99": 11.61630870819092,
    "max": 14.047297477722168
  },
  "target_normalization_mean": -25.585215168861424,
  "target_normalization_std": 49.7155473679296,
  "split_metrics": {
    "train": {
      "action_sample_count": 1102,
      "ranking_group_count": 72,
      "mae": 0.14240357181607122,
      "rmse": 0.1925382598647308,
      "pearson": 0.9542734146079748,
      "spearman": 0.7505912993375325,
      "argmax_accuracy": 0.7083333333333334,
      "recall_at_1": 0.7083333333333334,
      "recall_at_3": 0.9027777777777778,
      "recall_at_5": 0.9444444444444444,
      "exact_regret": {
        "mean": 0.016899380418989394,
        "p95": 0.12505440711975097,
        "max": 0.23301029205322266
      },
      "normalized_regret": {
        "mean": 0.03167249882820963,
        "p95": 0.19500035297977972,
        "max": 0.8806109427542981
      }
    },
    "validation": {
      "action_sample_count": 284,
      "ranking_group_count": 18,
      "mae": 4.885483381761224,
      "rmse": 6.825965381859383,
      "pearson": 0.11323548244638289,
      "spearman": 0.23393651569713692,
      "argmax_accuracy": 0.1111111111111111,
      "recall_at_1": 0.1111111111111111,
      "recall_at_3": 0.3333333333333333,
      "recall_at_5": 0.3888888888888889,
      "exact_regret": {
        "mean": 4.8295483059353295,
        "p95": 24.29629275798797,
        "max": 28.674465656280518
      },
      "normalized_regret": {
        "mean": 0.5641332767087459,
        "p95": 0.9999999999999587,
        "max": 0.9999999999999651
      }
    },
    "held_out": {
      "action_sample_count": 536,
      "ranking_group_count": 36,
      "mae": 17.931265918800346,
      "rmse": 20.441713950060162,
      "pearson": 0.23054779224960337,
      "spearman": 0.21113016713366697,
      "argmax_accuracy": 0.3055555555555556,
      "recall_at_1": 0.3055555555555556,
      "recall_at_3": 0.5277777777777778,
      "recall_at_5": 0.5833333333333334,
      "exact_regret": {
        "mean": 4.121613995792965,
        "p95": 16.879748702049255,
        "max": 25.863571286201477
      },
      "normalized_regret": {
        "mean": 0.43465835879449544,
        "p95": 0.9999999999948902,
        "max": 0.9999999999999014
      }
    }
  },
  "train_seconds": 19.858175100001972,
  "evaluation_seconds": 0.0,
  "status_detail": "PASS",
  "current_architecture_memorization": "PASS"
}
```

## C. Input aliasing / hidden-state insufficiency

```json
{
  "exact_duplicate_input_groups": 0,
  "conflicting_target_groups": 0,
  "max_target_spread_for_duplicate_input": 0.0,
  "exact_duplicate_summary": {
    "duplicate_input_groups": 0,
    "conflicting_target_groups": 0,
    "max_target_spread": 0.0,
    "mean_duplicate_group_size": null
  },
  "approximate_round5_summary": {
    "duplicate_input_groups": 0,
    "conflicting_target_groups": 0,
    "max_target_spread": 0.0,
    "mean_duplicate_group_size": null
  },
  "input_state_aliasing": "NOT_SUPPORTED",
  "grouping_method": "exact float32 bytes and rounded-to-5-decimal hash; no O(N^2) comparison"
}
```

## D. KEEP-relative target diagnostic

```json
{
  "current_residual_q_minus_v_train": {
    "mean": -25.585215168861424,
    "std": 49.7155473679296,
    "min": -155.53724670410156,
    "p1": -120.9492308807373,
    "p10": -120.0625213623047,
    "median": 3.0638864040374756,
    "p90": 10.19877576828003,
    "p99": 11.61630870819092,
    "max": 14.047297477722168
  },
  "keep_relative_delta_q_train": {
    "mean": 1.4755169474900094,
    "std": 24.12190381464606,
    "min": -113.74078750610352,
    "p1": -93.9606005859375,
    "p10": -3.8885838508605954,
    "median": -0.06153082847595215,
    "p90": 2.7742797851562497,
    "p99": 119.399466919899,
    "max": 128.84457778930664
  }
}
```

```json
{
  "delta_q_statistics_train": {
    "mean": 1.4755169474900094,
    "std": 24.12190381464606,
    "min": -113.74078750610352,
    "p1": -93.9606005859375,
    "p10": -3.8885838508605954,
    "median": -0.06153082847595215,
    "p90": 2.7742797851562497,
    "p99": 119.399466919899,
    "max": 128.84457778930664
  },
  "delta_q_sample_count": 1102
}
```

## E. Variant A — current residual baseline

```json
{
  "name": "VARIANT_A",
  "target_kind": "residual",
  "architecture": "cache_action_conditioned_residual_q_mlp",
  "architecture_version": 1,
  "fresh_initialization": true,
  "hidden_dim": 64,
  "max_epochs": 1000,
  "patience": 100,
  "early_stopping": true,
  "best_epoch": 152,
  "epochs_executed": 252,
  "best_validation_loss": 0.022428831085562706,
  "train_loss_initial": 0.9670089483261108,
  "train_loss_final": 0.004900948066885273,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.9670089483261108,
      "validation_loss": 0.35362276434898376
    },
    {
      "epoch": 2,
      "train_loss": 0.7670729094081454,
      "validation_loss": 0.025413045659661293
    },
    {
      "epoch": 3,
      "train_loss": 0.6397746470239427,
      "validation_loss": 0.15832889080047607
    },
    {
      "epoch": 4,
      "train_loss": 0.5567716525660621,
      "validation_loss": 0.08746954053640366
    },
    {
      "epoch": 5,
      "train_loss": 0.4485811922285292,
      "validation_loss": 0.03133615106344223
    },
    {
      "epoch": 6,
      "train_loss": 0.4411153594652812,
      "validation_loss": 0.15903356671333313
    },
    {
      "epoch": 7,
      "train_loss": 0.3677063352531857,
      "validation_loss": 0.31837305426597595
    },
    {
      "epoch": 8,
      "train_loss": 0.32618027925491333,
      "validation_loss": 0.070139579474926
    },
    {
      "epoch": 9,
      "train_loss": 0.32080403963724774,
      "validation_loss": 0.19248904287815094
    },
    {
      "epoch": 10,
      "train_loss": 0.25852569772137535,
      "validation_loss": 0.16170237958431244
    },
    {
      "epoch": 11,
      "train_loss": 0.2305315997865465,
      "validation_loss": 0.26879972219467163
    },
    {
      "epoch": 12,
      "train_loss": 0.23752027087741429,
      "validation_loss": 0.12995974719524384
    },
    {
      "epoch": 13,
      "train_loss": 0.22230189541975656,
      "validation_loss": 0.24869570136070251
    },
    {
      "epoch": 14,
      "train_loss": 0.20455306437280443,
      "validation_loss": 0.22096394002437592
    },
    {
      "epoch": 15,
      "train_loss": 0.19769109785556793,
      "validation_loss": 0.29512742161750793
    },
    {
      "epoch": 16,
      "train_loss": 0.19250747230317858,
      "validation_loss": 0.14681077003479004
    },
    {
      "epoch": 17,
      "train_loss": 0.19798420038488176,
      "validation_loss": 0.18950998783111572
    },
    {
      "epoch": 18,
      "train_loss": 0.19907689756817287,
      "validation_loss": 0.2213849574327469
    },
    {
      "epoch": 19,
      "train_loss": 0.1796902683046129,
      "validation_loss": 0.25904425978660583
    },
    {
      "epoch": 20,
      "train_loss": 0.17688763638337454,
      "validation_loss": 0.147054061293602
    },
    {
      "epoch": 21,
      "train_loss": 0.18305565085675982,
      "validation_loss": 0.3137674927711487
    },
    {
      "epoch": 22,
      "train_loss": 0.18512911515103447,
      "validation_loss": 0.18109115958213806
    },
    {
      "epoch": 23,
      "train_loss": 0.1724550930990113,
      "validation_loss": 0.20530188083648682
    },
    {
      "epoch": 24,
      "train_loss": 0.160256157318751,
      "validation_loss": 0.223084956407547
    },
    {
      "epoch": 25,
      "train_loss": 0.1605328776770168,
      "validation_loss": 0.21485081315040588
    },
    {
      "epoch": 26,
      "train_loss": 0.16829904086059994,
      "validation_loss": 0.13674010336399078
    },
    {
      "epoch": 27,
      "train_loss": 0.15627859036127725,
      "validation_loss": 0.14332029223442078
    },
    {
      "epoch": 28,
      "train_loss": 0.15462456478012931,
      "validation_loss": 0.15310800075531006
    },
    {
      "epoch": 29,
      "train_loss": 0.15146917601426443,
      "validation_loss": 0.1185954287648201
    },
    {
      "epoch": 30,
      "train_loss": 0.15615988191631106,
      "validation_loss": 0.3161616325378418
    },
    {
      "epoch": 31,
      "train_loss": 0.19299837946891785,
      "validation_loss": 0.11787240207195282
    },
    {
      "epoch": 32,
      "train_loss": 0.15251764158407846,
      "validation_loss": 0.11245955526828766
    },
    {
      "epoch": 33,
      "train_loss": 0.14524815645482805,
      "validation_loss": 0.16775690019130707
    },
    {
      "epoch": 34,
      "train_loss": 0.15962019066015878,
      "validation_loss": 0.08634321391582489
    },
    {
      "epoch": 35,
      "train_loss": 0.14814596705966526,
      "validation_loss": 0.09493532031774521
    },
    {
      "epoch": 36,
      "train_loss": 0.12770194808642069,
      "validation_loss": 0.09363730251789093
    },
    {
      "epoch": 37,
      "train_loss": 0.1349778100848198,
      "validation_loss": 0.14575321972370148
    },
    {
      "epoch": 38,
      "train_loss": 0.13885611626836988,
      "validation_loss": 0.06083245202898979
    },
    {
      "epoch": 39,
      "train_loss": 0.13295306513706842,
      "validation_loss": 0.11607497930526733
    },
    {
      "epoch": 40,
      "train_loss": 0.1231121313240793,
      "validation_loss": 0.050958309322595596
    },
    {
      "epoch": 41,
      "train_loss": 0.11444011661741468,
      "validation_loss": 0.12027420848608017
    },
    {
      "epoch": 42,
      "train_loss": 0.11058716475963593,
      "validation_loss": 0.07494081556797028
    },
    {
      "epoch": 43,
      "train_loss": 0.10425048569838206,
      "validation_loss": 0.10027194023132324
    },
    {
      "epoch": 44,
      "train_loss": 0.09571451817949612,
      "validation_loss": 0.05897791311144829
    },
    {
      "epoch": 45,
      "train_loss": 0.09663522243499756,
      "validation_loss": 0.11364001780748367
    },
    {
      "epoch": 46,
      "train_loss": 0.10284182677666347,
      "validation_loss": 0.041976701468229294
    },
    {
      "epoch": 47,
      "train_loss": 0.10865749998225106,
      "validation_loss": 0.07587414234876633
    },
    {
      "epoch": 48,
      "train_loss": 0.08189273832572831,
      "validation_loss": 0.08620152622461319
    },
    {
      "epoch": 49,
      "train_loss": 0.08144650980830193,
      "validation_loss": 0.044227804988622665
    },
    {
      "epoch": 50,
      "train_loss": 0.07454134316907989,
      "validation_loss": 0.06540564447641373
    },
    {
      "epoch": 51,
      "train_loss": 0.06805343594815996,
      "validation_loss": 0.05337921902537346
    },
    {
      "epoch": 52,
      "train_loss": 0.06729855595363511,
      "validation_loss": 0.056429293006658554
    },
    {
      "epoch": 53,
      "train_loss": 0.0661299570153157,
      "validation_loss": 0.06528060138225555
    },
    {
      "epoch": 54,
      "train_loss": 0.06774844312005573,
      "validation_loss": 0.03875619173049927
    },
    {
      "epoch": 55,
      "train_loss": 0.07193780193726222,
      "validation_loss": 0.06454223394393921
    },
    {
      "epoch": 56,
      "train_loss": 0.06278875097632408,
      "validation_loss": 0.04250553995370865
    },
    {
      "epoch": 57,
      "train_loss": 0.058146920055150986,
      "validation_loss": 0.04602403938770294
    },
    {
      "epoch": 58,
      "train_loss": 0.05013245820171303,
      "validation_loss": 0.03719839081168175
    },
    {
      "epoch": 59,
      "train_loss": 0.049704051266113915,
      "validation_loss": 0.050677213817834854
    },
    {
      "epoch": 60,
      "train_loss": 0.04728428481353654,
      "validation_loss": 0.038294583559036255
    },
    {
      "epoch": 61,
      "train_loss": 0.04812609197364913,
      "validation_loss": 0.04818790405988693
    },
    {
      "epoch": 62,
      "train_loss": 0.04410661322375139,
      "validation_loss": 0.048120759427547455
    },
    {
      "epoch": 63,
      "train_loss": 0.04721204853720135,
      "validation_loss": 0.0373498797416687
    },
    {
      "epoch": 64,
      "train_loss": 0.046985445337163076,
      "validation_loss": 0.04680008813738823
    },
    {
      "epoch": 65,
      "train_loss": 0.04524014352096452,
      "validation_loss": 0.04169740155339241
    },
    {
      "epoch": 66,
      "train_loss": 0.041195545345544815,
      "validation_loss": 0.0542319156229496
    },
    {
      "epoch": 67,
      "train_loss": 0.052820984895030655,
      "validation_loss": 0.03105718269944191
    },
    {
      "epoch": 68,
      "train_loss": 0.050937080962790385,
      "validation_loss": 0.04227865859866142
    },
    {
      "epoch": 69,
      "train_loss": 0.04093843595021301,
      "validation_loss": 0.029319360852241516
    },
    {
      "epoch": 70,
      "train_loss": 0.046564868133929044,
      "validation_loss": 0.045132432132959366
    },
    {
      "epoch": 71,
      "train_loss": 0.04135860544111994,
      "validation_loss": 0.03211641311645508
    },
    {
      "epoch": 72,
      "train_loss": 0.054086744578348264,
      "validation_loss": 0.03520115092396736
    },
    {
      "epoch": 73,
      "train_loss": 0.05222185100946161,
      "validation_loss": 0.03333277627825737
    },
    {
      "epoch": 74,
      "train_loss": 0.03812821002470122,
      "validation_loss": 0.03424827381968498
    },
    {
      "epoch": 75,
      "train_loss": 0.03666503189338578,
      "validation_loss": 0.032356441020965576
    },
    {
      "epoch": 76,
      "train_loss": 0.03155734659069114,
      "validation_loss": 0.03131580725312233
    },
    {
      "epoch": 77,
      "train_loss": 0.031430195603105754,
      "validation_loss": 0.03288673609495163
    },
    {
      "epoch": 78,
      "train_loss": 0.03148655821051863,
      "validation_loss": 0.027187561616301537
    },
    {
      "epoch": 79,
      "train_loss": 0.031615350395441055,
      "validation_loss": 0.035202398896217346
    },
    {
      "epoch": 80,
      "train_loss": 0.03153836768534449,
      "validation_loss": 0.029453210532665253
    },
    {
      "epoch": 81,
      "train_loss": 0.02642695088353422,
      "validation_loss": 0.02834218367934227
    },
    {
      "epoch": 82,
      "train_loss": 0.025865680641598172,
      "validation_loss": 0.033136509358882904
    },
    {
      "epoch": 83,
      "train_loss": 0.024947799742221832,
      "validation_loss": 0.03139882907271385
    },
    {
      "epoch": 84,
      "train_loss": 0.024123871905936137,
      "validation_loss": 0.027845166623592377
    },
    {
      "epoch": 85,
      "train_loss": 0.027370489098959498,
      "validation_loss": 0.02640870213508606
    },
    {
      "epoch": 86,
      "train_loss": 0.026989666124184925,
      "validation_loss": 0.028795648366212845
    },
    {
      "epoch": 87,
      "train_loss": 0.023500378140144877,
      "validation_loss": 0.02643461339175701
    },
    {
      "epoch": 88,
      "train_loss": 0.021930328156385157,
      "validation_loss": 0.030065033584833145
    },
    {
      "epoch": 89,
      "train_loss": 0.022925008709232014,
      "validation_loss": 0.026681898161768913
    },
    {
      "epoch": 90,
      "train_loss": 0.02071332114024295,
      "validation_loss": 0.027418989688158035
    },
    {
      "epoch": 91,
      "train_loss": 0.022573464653558202,
      "validation_loss": 0.02862120419740677
    },
    {
      "epoch": 92,
      "train_loss": 0.021998967561456893,
      "validation_loss": 0.026460086926817894
    },
    {
      "epoch": 93,
      "train_loss": 0.021163043255607288,
      "validation_loss": 0.025587022304534912
    },
    {
      "epoch": 94,
      "train_loss": 0.02141553981022702,
      "validation_loss": 0.02702009119093418
    },
    {
      "epoch": 95,
      "train_loss": 0.020925541408360004,
      "validation_loss": 0.02615225873887539
    },
    {
      "epoch": 96,
      "train_loss": 0.021114869353671867,
      "validation_loss": 0.030352119356393814
    },
    {
      "epoch": 97,
      "train_loss": 0.019965145426491897,
      "validation_loss": 0.02578371949493885
    },
    {
      "epoch": 98,
      "train_loss": 0.019007984548807144,
      "validation_loss": 0.026725012809038162
    },
    {
      "epoch": 99,
      "train_loss": 0.01937240579475959,
      "validation_loss": 0.029242519289255142
    },
    {
      "epoch": 100,
      "train_loss": 0.018589054958687887,
      "validation_loss": 0.025174932554364204
    },
    {
      "epoch": 101,
      "train_loss": 0.018848366828428373,
      "validation_loss": 0.02372284233570099
    },
    {
      "epoch": 102,
      "train_loss": 0.01975679263058636,
      "validation_loss": 0.0394388809800148
    },
    {
      "epoch": 103,
      "train_loss": 0.019532447163429525,
      "validation_loss": 0.026159893721342087
    },
    {
      "epoch": 104,
      "train_loss": 0.020178259143398866,
      "validation_loss": 0.023954348638653755
    },
    {
      "epoch": 105,
      "train_loss": 0.01863332682599624,
      "validation_loss": 0.03271856531500816
    },
    {
      "epoch": 106,
      "train_loss": 0.02001560479402542,
      "validation_loss": 0.03358122333884239
    },
    {
      "epoch": 107,
      "train_loss": 0.017364645790722635,
      "validation_loss": 0.025504326447844505
    },
    {
      "epoch": 108,
      "train_loss": 0.018319587430192366,
      "validation_loss": 0.026582250371575356
    },
    {
      "epoch": 109,
      "train_loss": 0.021371898448301688,
      "validation_loss": 0.037354666739702225
    },
    {
      "epoch": 110,
      "train_loss": 0.0202659766914116,
      "validation_loss": 0.025420837104320526
    },
    {
      "epoch": 111,
      "train_loss": 0.023180602221853204,
      "validation_loss": 0.02649668976664543
    },
    {
      "epoch": 112,
      "train_loss": 0.01886789407581091,
      "validation_loss": 0.03108963556587696
    },
    {
      "epoch": 113,
      "train_loss": 0.01868450227710936,
      "validation_loss": 0.027259360998868942
    },
    {
      "epoch": 114,
      "train_loss": 0.0164801732947429,
      "validation_loss": 0.02539134956896305
    },
    {
      "epoch": 115,
      "train_loss": 0.01548540530105432,
      "validation_loss": 0.03423655033111572
    },
    {
      "epoch": 116,
      "train_loss": 0.020822276671727497,
      "validation_loss": 0.02705441229045391
    },
    {
      "epoch": 117,
      "train_loss": 0.015963049605488777,
      "validation_loss": 0.024855749681591988
    },
    {
      "epoch": 118,
      "train_loss": 0.014386724370221296,
      "validation_loss": 0.031137550249695778
    },
    {
      "epoch": 119,
      "train_loss": 0.013235284946858883,
      "validation_loss": 0.02715263143181801
    },
    {
      "epoch": 120,
      "train_loss": 0.013511982435981432,
      "validation_loss": 0.022481977939605713
    },
    {
      "epoch": 121,
      "train_loss": 0.013734043886264166,
      "validation_loss": 0.028321001678705215
    },
    {
      "epoch": 122,
      "train_loss": 0.013703956993089782,
      "validation_loss": 0.027017641812562943
    },
    {
      "epoch": 123,
      "train_loss": 0.013630891839663187,
      "validation_loss": 0.028332838788628578
    },
    {
      "epoch": 124,
      "train_loss": 0.015523516366051303,
      "validation_loss": 0.028620555996894836
    },
    {
      "epoch": 125,
      "train_loss": 0.013794054070280658,
      "validation_loss": 0.02629798837006092
    },
    {
      "epoch": 126,
      "train_loss": 0.013682605595224433,
      "validation_loss": 0.02603638917207718
    },
    {
      "epoch": 127,
      "train_loss": 0.015767277632322576,
      "validation_loss": 0.025394905358552933
    },
    {
      "epoch": 128,
      "train_loss": 0.01792887256791194,
      "validation_loss": 0.026349246501922607
    },
    {
      "epoch": 129,
      "train_loss": 0.014814612352185778,
      "validation_loss": 0.03147212043404579
    },
    {
      "epoch": 130,
      "train_loss": 0.012178102663407723,
      "validation_loss": 0.028371142223477364
    },
    {
      "epoch": 131,
      "train_loss": 0.012264126704798805,
      "validation_loss": 0.02453789860010147
    },
    {
      "epoch": 132,
      "train_loss": 0.01473305436472098,
      "validation_loss": 0.026366889476776123
    },
    {
      "epoch": 133,
      "train_loss": 0.013892419946690401,
      "validation_loss": 0.027465593069791794
    },
    {
      "epoch": 134,
      "train_loss": 0.01196768455621269,
      "validation_loss": 0.027443954721093178
    },
    {
      "epoch": 135,
      "train_loss": 0.01107034252749549,
      "validation_loss": 0.026473533362150192
    },
    {
      "epoch": 136,
      "train_loss": 0.01176156915931238,
      "validation_loss": 0.027311207726597786
    },
    {
      "epoch": 137,
      "train_loss": 0.011401556535727449,
      "validation_loss": 0.024910930544137955
    },
    {
      "epoch": 138,
      "train_loss": 0.01157102764894565,
      "validation_loss": 0.03558076173067093
    },
    {
      "epoch": 139,
      "train_loss": 0.012569615099993017,
      "validation_loss": 0.03109043464064598
    },
    {
      "epoch": 140,
      "train_loss": 0.011494936162812842,
      "validation_loss": 0.026267271488904953
    },
    {
      "epoch": 141,
      "train_loss": 0.011688815067625709,
      "validation_loss": 0.027015792205929756
    },
    {
      "epoch": 142,
      "train_loss": 0.011978824622929096,
      "validation_loss": 0.029226528480648994
    },
    {
      "epoch": 143,
      "train_loss": 0.010541717263145579,
      "validation_loss": 0.02769593708217144
    },
    {
      "epoch": 144,
      "train_loss": 0.010689213613255156,
      "validation_loss": 0.026517553254961967
    },
    {
      "epoch": 145,
      "train_loss": 0.013049246898541847,
      "validation_loss": 0.03145243227481842
    },
    {
      "epoch": 146,
      "train_loss": 0.012474879260278411,
      "validation_loss": 0.025977127254009247
    },
    {
      "epoch": 147,
      "train_loss": 0.012849756671736637,
      "validation_loss": 0.030765149742364883
    },
    {
      "epoch": 148,
      "train_loss": 0.014532870509558253,
      "validation_loss": 0.029509125277400017
    },
    {
      "epoch": 149,
      "train_loss": 0.012505971723132663,
      "validation_loss": 0.02720305696129799
    },
    {
      "epoch": 150,
      "train_loss": 0.01198006711072392,
      "validation_loss": 0.025911135599017143
    },
    {
      "epoch": 151,
      "train_loss": 0.011442326733635532,
      "validation_loss": 0.026335950940847397
    },
    {
      "epoch": 152,
      "train_loss": 0.011960472290714582,
      "validation_loss": 0.022428831085562706
    },
    {
      "epoch": 153,
      "train_loss": 0.01134645887133148,
      "validation_loss": 0.02812577225267887
    },
    {
      "epoch": 154,
      "train_loss": 0.011507675051689148,
      "validation_loss": 0.02824934385716915
    },
    {
      "epoch": 155,
      "train_loss": 0.008445735451661877,
      "validation_loss": 0.026723036542534828
    },
    {
      "epoch": 156,
      "train_loss": 0.008742169477045536,
      "validation_loss": 0.0252167247235775
    },
    {
      "epoch": 157,
      "train_loss": 0.009532255586236715,
      "validation_loss": 0.024377860128879547
    },
    {
      "epoch": 158,
      "train_loss": 0.009314872014025847,
      "validation_loss": 0.02702820487320423
    },
    {
      "epoch": 159,
      "train_loss": 0.008982425245145956,
      "validation_loss": 0.023468267172574997
    },
    {
      "epoch": 160,
      "train_loss": 0.008291125970168246,
      "validation_loss": 0.024511361494660378
    },
    {
      "epoch": 161,
      "train_loss": 0.008892763561258713,
      "validation_loss": 0.03342435881495476
    },
    {
      "epoch": 162,
      "train_loss": 0.00840774199201001,
      "validation_loss": 0.027242667973041534
    },
    {
      "epoch": 163,
      "train_loss": 0.0083930897526443,
      "validation_loss": 0.02536606602370739
    },
    {
      "epoch": 164,
      "train_loss": 0.008382867142144177,
      "validation_loss": 0.024384604766964912
    },
    {
      "epoch": 165,
      "train_loss": 0.009846008589698209,
      "validation_loss": 0.027751022949814796
    },
    {
      "epoch": 166,
      "train_loss": 0.009435781329456303,
      "validation_loss": 0.028035810217261314
    },
    {
      "epoch": 167,
      "train_loss": 0.010628958077480396,
      "validation_loss": 0.028226500377058983
    },
    {
      "epoch": 168,
      "train_loss": 0.010668049462967448,
      "validation_loss": 0.03649447485804558
    },
    {
      "epoch": 169,
      "train_loss": 0.010145367537107732,
      "validation_loss": 0.031062832102179527
    },
    {
      "epoch": 170,
      "train_loss": 0.008340941224661138,
      "validation_loss": 0.02952173352241516
    },
    {
      "epoch": 171,
      "train_loss": 0.00866586042361127,
      "validation_loss": 0.028132563456892967
    },
    {
      "epoch": 172,
      "train_loss": 0.009169910258303085,
      "validation_loss": 0.030894998461008072
    },
    {
      "epoch": 173,
      "train_loss": 0.010659020652787553,
      "validation_loss": 0.024601824581623077
    },
    {
      "epoch": 174,
      "train_loss": 0.009253218381976088,
      "validation_loss": 0.02885015867650509
    },
    {
      "epoch": 175,
      "train_loss": 0.010350958340697818,
      "validation_loss": 0.030807962641119957
    },
    {
      "epoch": 176,
      "train_loss": 0.010632520489808586,
      "validation_loss": 0.028850648552179337
    },
    {
      "epoch": 177,
      "train_loss": 0.009313319757994678,
      "validation_loss": 0.03082125633955002
    },
    {
      "epoch": 178,
      "train_loss": 0.008433038006640144,
      "validation_loss": 0.02740810625255108
    },
    {
      "epoch": 179,
      "train_loss": 0.006988023666457998,
      "validation_loss": 0.028307421132922173
    },
    {
      "epoch": 180,
      "train_loss": 0.007510322249597973,
      "validation_loss": 0.028054993599653244
    },
    {
      "epoch": 181,
      "train_loss": 0.007566910030113326,
      "validation_loss": 0.030154362320899963
    },
    {
      "epoch": 182,
      "train_loss": 0.006832044830338823,
      "validation_loss": 0.03121439926326275
    },
    {
      "epoch": 183,
      "train_loss": 0.007602806616988447,
      "validation_loss": 0.028221845626831055
    },
    {
      "epoch": 184,
      "train_loss": 0.0063684603406323325,
      "validation_loss": 0.027698976919054985
    },
    {
      "epoch": 185,
      "train_loss": 0.006513503664690588,
      "validation_loss": 0.030648022890090942
    },
    {
      "epoch": 186,
      "train_loss": 0.006809731583214468,
      "validation_loss": 0.030810242518782616
    },
    {
      "epoch": 187,
      "train_loss": 0.008158454826722542,
      "validation_loss": 0.03257155418395996
    },
    {
      "epoch": 188,
      "train_loss": 0.007570154913183715,
      "validation_loss": 0.026057472452521324
    },
    {
      "epoch": 189,
      "train_loss": 0.006674864639838536,
      "validation_loss": 0.026365455240011215
    },
    {
      "epoch": 190,
      "train_loss": 0.007259312137547467,
      "validation_loss": 0.026671715080738068
    },
    {
      "epoch": 191,
      "train_loss": 0.0061309670305086505,
      "validation_loss": 0.025810087099671364
    },
    {
      "epoch": 192,
      "train_loss": 0.006097587498111857,
      "validation_loss": 0.031928546726703644
    },
    {
      "epoch": 193,
      "train_loss": 0.005958445680638154,
      "validation_loss": 0.028423795476555824
    },
    {
      "epoch": 194,
      "train_loss": 0.006172754957030217,
      "validation_loss": 0.027385344728827477
    },
    {
      "epoch": 195,
      "train_loss": 0.006316905323829915,
      "validation_loss": 0.023143885657191277
    },
    {
      "epoch": 196,
      "train_loss": 0.006731854829316338,
      "validation_loss": 0.02674773521721363
    },
    {
      "epoch": 197,
      "train_loss": 0.007426472846418619,
      "validation_loss": 0.027852235361933708
    },
    {
      "epoch": 198,
      "train_loss": 0.0077802410038808984,
      "validation_loss": 0.0313035286962986
    },
    {
      "epoch": 199,
      "train_loss": 0.00644134768905739,
      "validation_loss": 0.02981002815067768
    },
    {
      "epoch": 200,
      "train_loss": 0.006436162524753147,
      "validation_loss": 0.029117442667484283
    },
    {
      "epoch": 201,
      "train_loss": 0.00573023920878768,
      "validation_loss": 0.025522595271468163
    },
    {
      "epoch": 202,
      "train_loss": 0.00810047321849399,
      "validation_loss": 0.02814379148185253
    },
    {
      "epoch": 203,
      "train_loss": 0.008267865723205937,
      "validation_loss": 0.026764318346977234
    },
    {
      "epoch": 204,
      "train_loss": 0.006317234908541043,
      "validation_loss": 0.025526586920022964
    },
    {
      "epoch": 205,
      "train_loss": 0.006324374582618475,
      "validation_loss": 0.025625383481383324
    },
    {
      "epoch": 206,
      "train_loss": 0.007340948097407818,
      "validation_loss": 0.028181632980704308
    },
    {
      "epoch": 207,
      "train_loss": 0.00602115325940152,
      "validation_loss": 0.03196064010262489
    },
    {
      "epoch": 208,
      "train_loss": 0.005504426675745183,
      "validation_loss": 0.029281143099069595
    },
    {
      "epoch": 209,
      "train_loss": 0.006208185003035598,
      "validation_loss": 0.028266647830605507
    },
    {
      "epoch": 210,
      "train_loss": 0.005944869584507412,
      "validation_loss": 0.028616879135370255
    },
    {
      "epoch": 211,
      "train_loss": 0.00615455383538372,
      "validation_loss": 0.03098190948367119
    },
    {
      "epoch": 212,
      "train_loss": 0.00541230793007546,
      "validation_loss": 0.030725378543138504
    },
    {
      "epoch": 213,
      "train_loss": 0.005748634252490269,
      "validation_loss": 0.031063096597790718
    },
    {
      "epoch": 214,
      "train_loss": 0.005528693863501151,
      "validation_loss": 0.03453763201832771
    },
    {
      "epoch": 215,
      "train_loss": 0.005211868866657217,
      "validation_loss": 0.03203660622239113
    },
    {
      "epoch": 216,
      "train_loss": 0.005284884504766928,
      "validation_loss": 0.02760857343673706
    },
    {
      "epoch": 217,
      "train_loss": 0.005059840726769633,
      "validation_loss": 0.0323777012526989
    },
    {
      "epoch": 218,
      "train_loss": 0.005144509163478183,
      "validation_loss": 0.031121274456381798
    },
    {
      "epoch": 219,
      "train_loss": 0.005142968096252944,
      "validation_loss": 0.02917367033660412
    },
    {
      "epoch": 220,
      "train_loss": 0.004642987209889624,
      "validation_loss": 0.03373953700065613
    },
    {
      "epoch": 221,
      "train_loss": 0.0049651936990105444,
      "validation_loss": 0.03373124077916145
    },
    {
      "epoch": 222,
      "train_loss": 0.004874922438628144,
      "validation_loss": 0.03132123500108719
    },
    {
      "epoch": 223,
      "train_loss": 0.005345427068985171,
      "validation_loss": 0.029789013788104057
    },
    {
      "epoch": 224,
      "train_loss": 0.00636172325660785,
      "validation_loss": 0.025455327704548836
    },
    {
      "epoch": 225,
      "train_loss": 0.006062461839367946,
      "validation_loss": 0.02806900069117546
    },
    {
      "epoch": 226,
      "train_loss": 0.00490169747111698,
      "validation_loss": 0.027043098583817482
    },
    {
      "epoch": 227,
      "train_loss": 0.005362012656405568,
      "validation_loss": 0.029928959906101227
    },
    {
      "epoch": 228,
      "train_loss": 0.005000171199854877,
      "validation_loss": 0.033987268805503845
    },
    {
      "epoch": 229,
      "train_loss": 0.004850542431490289,
      "validation_loss": 0.032725583761930466
    },
    {
      "epoch": 230,
      "train_loss": 0.0056548642460256815,
      "validation_loss": 0.032498057931661606
    },
    {
      "epoch": 231,
      "train_loss": 0.005438369336641497,
      "validation_loss": 0.029446769505739212
    },
    {
      "epoch": 232,
      "train_loss": 0.004594932853554686,
      "validation_loss": 0.0282414760440588
    },
    {
      "epoch": 233,
      "train_loss": 0.0048352516411493225,
      "validation_loss": 0.03172539174556732
    },
    {
      "epoch": 234,
      "train_loss": 0.004577637287891573,
      "validation_loss": 0.03585929796099663
    },
    {
      "epoch": 235,
      "train_loss": 0.004987736465409398,
      "validation_loss": 0.03298478201031685
    },
    {
      "epoch": 236,
      "train_loss": 0.005295567628410127,
      "validation_loss": 0.0380423441529274
    },
    {
      "epoch": 237,
      "train_loss": 0.008272686455812719,
      "validation_loss": 0.03260671719908714
    },
    {
      "epoch": 238,
      "train_loss": 0.008072988906254372,
      "validation_loss": 0.029402021318674088
    },
    {
      "epoch": 239,
      "train_loss": 0.006656998639098472,
      "validation_loss": 0.026665566489100456
    },
    {
      "epoch": 240,
      "train_loss": 0.005602903871072663,
      "validation_loss": 0.028451815247535706
    },
    {
      "epoch": 241,
      "train_loss": 0.005497210969527562,
      "validation_loss": 0.030515817925333977
    },
    {
      "epoch": 242,
      "train_loss": 0.005311663624727064,
      "validation_loss": 0.028902752324938774
    },
    {
      "epoch": 243,
      "train_loss": 0.005695872733162509,
      "validation_loss": 0.02492431364953518
    },
    {
      "epoch": 244,
      "train_loss": 0.008051754311761923,
      "validation_loss": 0.0276078712195158
    },
    {
      "epoch": 245,
      "train_loss": 0.00908012507069442,
      "validation_loss": 0.027780933305621147
    },
    {
      "epoch": 246,
      "train_loss": 0.007535926764830947,
      "validation_loss": 0.02950962446630001
    },
    {
      "epoch": 247,
      "train_loss": 0.005254327630003293,
      "validation_loss": 0.03016042336821556
    },
    {
      "epoch": 248,
      "train_loss": 0.004344782766161693,
      "validation_loss": 0.029212789610028267
    },
    {
      "epoch": 249,
      "train_loss": 0.0056701479479670525,
      "validation_loss": 0.029441531747579575
    },
    {
      "epoch": 250,
      "train_loss": 0.00621602893806994,
      "validation_loss": 0.03154712915420532
    },
    {
      "epoch": 251,
      "train_loss": 0.006146849495255285,
      "validation_loss": 0.03419168293476105
    },
    {
      "epoch": 252,
      "train_loss": 0.004900948066885273,
      "validation_loss": 0.03108145482838154
    }
  ],
  "target_statistics_train_only": {
    "mean": -25.585215168861424,
    "std": 49.7155473679296,
    "min": -155.53724670410156,
    "p1": -120.9492308807373,
    "p10": -120.0625213623047,
    "median": 3.0638864040374756,
    "p90": 10.19877576828003,
    "p99": 11.61630870819092,
    "max": 14.047297477722168
  },
  "target_normalization_mean": -25.585215168861424,
  "target_normalization_std": 49.7155473679296,
  "split_metrics": {
    "train": {
      "action_sample_count": 1102,
      "ranking_group_count": 72,
      "mae": 3.3546077433655555,
      "rmse": 4.8752771434988835,
      "pearson": 0.6508095861080062,
      "spearman": 0.43575305690365274,
      "argmax_accuracy": 0.4305555555555556,
      "recall_at_1": 0.4305555555555556,
      "recall_at_3": 0.625,
      "recall_at_5": 0.7083333333333334,
      "exact_regret": {
        "mean": 0.664439492755466,
        "p95": 3.097502112388612,
        "max": 6.359392881393433
      },
      "normalized_regret": {
        "mean": 0.1679181406306879,
        "p95": 0.9430102938047542,
        "max": 0.9994527171921282
      }
    },
    "validation": {
      "action_sample_count": 284,
      "ranking_group_count": 18,
      "mae": 4.602222112550914,
      "rmse": 7.445528837036096,
      "pearson": 0.013214157929465821,
      "spearman": 0.15393236169543834,
      "argmax_accuracy": 0.05555555555555555,
      "recall_at_1": 0.05555555555555555,
      "recall_at_3": 0.2222222222222222,
      "recall_at_5": 0.3888888888888889,
      "exact_regret": {
        "mean": 3.187376936276754,
        "p95": 10.854813152551635,
        "max": 19.642959594726562
      },
      "normalized_regret": {
        "mean": 0.5052603968134303,
        "p95": 0.9778936127357288,
        "max": 0.9999999999941874
      }
    },
    "held_out": {
      "action_sample_count": 536,
      "ranking_group_count": 36,
      "mae": 18.391964321226318,
      "rmse": 21.598069973017136,
      "pearson": 0.12887458457971812,
      "spearman": 0.25100663344017093,
      "argmax_accuracy": 0.2222222222222222,
      "recall_at_1": 0.2222222222222222,
      "recall_at_3": 0.3888888888888889,
      "recall_at_5": 0.5,
      "exact_regret": {
        "mean": 6.055315361979107,
        "p95": 26.02702099084854,
        "max": 32.5841269493103
      },
      "normalized_regret": {
        "mean": 0.4729334458188851,
        "p95": 0.9999999999981409,
        "max": 0.9999999999999014
      }
    }
  },
  "train_seconds": 2.5645752000273205,
  "evaluation_seconds": 0.0,
  "fit_status": "FAIL"
}
```

## F. Variant B — KEEP-relative delta Q

```json
{
  "name": "VARIANT_B",
  "target_kind": "delta",
  "architecture": "cache_action_conditioned_residual_q_mlp",
  "architecture_version": 1,
  "fresh_initialization": true,
  "hidden_dim": 64,
  "max_epochs": 1000,
  "patience": 100,
  "early_stopping": true,
  "best_epoch": 160,
  "epochs_executed": 260,
  "best_validation_loss": 0.024521654471755028,
  "train_loss_initial": 0.9831908808814155,
  "train_loss_final": 0.0144692479322354,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.9831908808814155,
      "validation_loss": 0.04419422149658203
    },
    {
      "epoch": 2,
      "train_loss": 0.9194123248259226,
      "validation_loss": 0.03968554735183716
    },
    {
      "epoch": 3,
      "train_loss": 0.8981418742073907,
      "validation_loss": 0.0649832934141159
    },
    {
      "epoch": 4,
      "train_loss": 0.8236048287815518,
      "validation_loss": 0.08548204600811005
    },
    {
      "epoch": 5,
      "train_loss": 0.7810168100727929,
      "validation_loss": 0.1513948142528534
    },
    {
      "epoch": 6,
      "train_loss": 0.7662163707945082,
      "validation_loss": 0.08424732834100723
    },
    {
      "epoch": 7,
      "train_loss": 0.7217073606120216,
      "validation_loss": 0.09151095896959305
    },
    {
      "epoch": 8,
      "train_loss": 0.7918280296855502,
      "validation_loss": 0.20609533786773682
    },
    {
      "epoch": 9,
      "train_loss": 0.7841286791695489,
      "validation_loss": 0.1538621485233307
    },
    {
      "epoch": 10,
      "train_loss": 0.7627776794963412,
      "validation_loss": 0.12371104955673218
    },
    {
      "epoch": 11,
      "train_loss": 0.7362653745545281,
      "validation_loss": 0.12363877147436142
    },
    {
      "epoch": 12,
      "train_loss": 0.7016438477569156,
      "validation_loss": 0.13257169723510742
    },
    {
      "epoch": 13,
      "train_loss": 0.7015659080611335,
      "validation_loss": 0.13027730584144592
    },
    {
      "epoch": 14,
      "train_loss": 0.710696193906996,
      "validation_loss": 0.1163250058889389
    },
    {
      "epoch": 15,
      "train_loss": 0.7049183679951562,
      "validation_loss": 0.1339889019727707
    },
    {
      "epoch": 16,
      "train_loss": 0.7168202665117052,
      "validation_loss": 0.10363701730966568
    },
    {
      "epoch": 17,
      "train_loss": 0.7137355870670743,
      "validation_loss": 0.13269755244255066
    },
    {
      "epoch": 18,
      "train_loss": 0.7029659284485711,
      "validation_loss": 0.1265789121389389
    },
    {
      "epoch": 19,
      "train_loss": 0.6869344181484647,
      "validation_loss": 0.11249314248561859
    },
    {
      "epoch": 20,
      "train_loss": 0.6681258645322588,
      "validation_loss": 0.09978173673152924
    },
    {
      "epoch": 21,
      "train_loss": 0.6586723923683167,
      "validation_loss": 0.13781331479549408
    },
    {
      "epoch": 22,
      "train_loss": 0.6633466018570794,
      "validation_loss": 0.10112477838993073
    },
    {
      "epoch": 23,
      "train_loss": 0.6646389100286696,
      "validation_loss": 0.11002898216247559
    },
    {
      "epoch": 24,
      "train_loss": 0.6379177769025167,
      "validation_loss": 0.11102408170700073
    },
    {
      "epoch": 25,
      "train_loss": 0.6306101282437643,
      "validation_loss": 0.09956081211566925
    },
    {
      "epoch": 26,
      "train_loss": 0.633664415942298,
      "validation_loss": 0.09736112505197525
    },
    {
      "epoch": 27,
      "train_loss": 0.6175104280312856,
      "validation_loss": 0.09235773980617523
    },
    {
      "epoch": 28,
      "train_loss": 0.6122303472624885,
      "validation_loss": 0.08576405793428421
    },
    {
      "epoch": 29,
      "train_loss": 0.5958974725670285,
      "validation_loss": 0.09019584953784943
    },
    {
      "epoch": 30,
      "train_loss": 0.6224244832992554,
      "validation_loss": 0.12085971236228943
    },
    {
      "epoch": 31,
      "train_loss": 0.604236920674642,
      "validation_loss": 0.07321275770664215
    },
    {
      "epoch": 32,
      "train_loss": 0.5535100118981467,
      "validation_loss": 0.07976479083299637
    },
    {
      "epoch": 33,
      "train_loss": 0.5689828428957198,
      "validation_loss": 0.07306500524282455
    },
    {
      "epoch": 34,
      "train_loss": 0.6430505712827047,
      "validation_loss": 0.15026721358299255
    },
    {
      "epoch": 35,
      "train_loss": 0.6009170975950029,
      "validation_loss": 0.0725688561797142
    },
    {
      "epoch": 36,
      "train_loss": 0.5362211200926039,
      "validation_loss": 0.04507860168814659
    },
    {
      "epoch": 37,
      "train_loss": 0.52144788702329,
      "validation_loss": 0.05487977713346481
    },
    {
      "epoch": 38,
      "train_loss": 0.496644311481052,
      "validation_loss": 0.06156916171312332
    },
    {
      "epoch": 39,
      "train_loss": 0.4961472931835387,
      "validation_loss": 0.048805054277181625
    },
    {
      "epoch": 40,
      "train_loss": 0.519511365228229,
      "validation_loss": 0.047092411667108536
    },
    {
      "epoch": 41,
      "train_loss": 0.5147184398439195,
      "validation_loss": 0.043240297585725784
    },
    {
      "epoch": 42,
      "train_loss": 0.4821451140774621,
      "validation_loss": 0.03912637010216713
    },
    {
      "epoch": 43,
      "train_loss": 0.48366155558162266,
      "validation_loss": 0.03478279709815979
    },
    {
      "epoch": 44,
      "train_loss": 0.4880810048845079,
      "validation_loss": 0.03870739787817001
    },
    {
      "epoch": 45,
      "train_loss": 0.4496132532755534,
      "validation_loss": 0.0401863232254982
    },
    {
      "epoch": 46,
      "train_loss": 0.41936874555216896,
      "validation_loss": 0.038634318858385086
    },
    {
      "epoch": 47,
      "train_loss": 0.44161095718542737,
      "validation_loss": 0.04043643921613693
    },
    {
      "epoch": 48,
      "train_loss": 0.437043954928716,
      "validation_loss": 0.03160782903432846
    },
    {
      "epoch": 49,
      "train_loss": 0.40530719525284237,
      "validation_loss": 0.03904683142900467
    },
    {
      "epoch": 50,
      "train_loss": 0.3912799490822686,
      "validation_loss": 0.029986968263983727
    },
    {
      "epoch": 51,
      "train_loss": 0.38818735215399003,
      "validation_loss": 0.0419691801071167
    },
    {
      "epoch": 52,
      "train_loss": 0.3803401125801934,
      "validation_loss": 0.028192903846502304
    },
    {
      "epoch": 53,
      "train_loss": 0.3632647030883365,
      "validation_loss": 0.04142965376377106
    },
    {
      "epoch": 54,
      "train_loss": 0.526304809583558,
      "validation_loss": 0.05722867324948311
    },
    {
      "epoch": 55,
      "train_loss": 0.43472614884376526,
      "validation_loss": 0.028383638709783554
    },
    {
      "epoch": 56,
      "train_loss": 0.3555571999814775,
      "validation_loss": 0.025474444031715393
    },
    {
      "epoch": 57,
      "train_loss": 0.3595962193277147,
      "validation_loss": 0.02637314796447754
    },
    {
      "epoch": 58,
      "train_loss": 0.3564937180942959,
      "validation_loss": 0.026880301535129547
    },
    {
      "epoch": 59,
      "train_loss": 0.335425219602055,
      "validation_loss": 0.03985362499952316
    },
    {
      "epoch": 60,
      "train_loss": 0.3827187567949295,
      "validation_loss": 0.02786029875278473
    },
    {
      "epoch": 61,
      "train_loss": 0.4051140265332328,
      "validation_loss": 0.0312101598829031
    },
    {
      "epoch": 62,
      "train_loss": 0.5462850845522351,
      "validation_loss": 0.04323802888393402
    },
    {
      "epoch": 63,
      "train_loss": 0.5235317548116049,
      "validation_loss": 0.02722499892115593
    },
    {
      "epoch": 64,
      "train_loss": 0.39043448699845207,
      "validation_loss": 0.02759014256298542
    },
    {
      "epoch": 65,
      "train_loss": 0.32966773211956024,
      "validation_loss": 0.026146508753299713
    },
    {
      "epoch": 66,
      "train_loss": 0.316366551650895,
      "validation_loss": 0.027860701084136963
    },
    {
      "epoch": 67,
      "train_loss": 0.29601344962914783,
      "validation_loss": 0.031393345445394516
    },
    {
      "epoch": 68,
      "train_loss": 0.2607765288816558,
      "validation_loss": 0.026665201410651207
    },
    {
      "epoch": 69,
      "train_loss": 0.25349337524837917,
      "validation_loss": 0.026153558865189552
    },
    {
      "epoch": 70,
      "train_loss": 0.23675608138243356,
      "validation_loss": 0.02690216898918152
    },
    {
      "epoch": 71,
      "train_loss": 0.2262998107406828,
      "validation_loss": 0.029193038120865822
    },
    {
      "epoch": 72,
      "train_loss": 0.22099385327763027,
      "validation_loss": 0.025795569643378258
    },
    {
      "epoch": 73,
      "train_loss": 0.2205487150284979,
      "validation_loss": 0.025531644001603127
    },
    {
      "epoch": 74,
      "train_loss": 0.25957557807366055,
      "validation_loss": 0.031803421676158905
    },
    {
      "epoch": 75,
      "train_loss": 0.21649773915608725,
      "validation_loss": 0.02894885465502739
    },
    {
      "epoch": 76,
      "train_loss": 0.18156751038299668,
      "validation_loss": 0.025940921157598495
    },
    {
      "epoch": 77,
      "train_loss": 0.21010457972685495,
      "validation_loss": 0.028551457449793816
    },
    {
      "epoch": 78,
      "train_loss": 0.19497708645131853,
      "validation_loss": 0.026726361364126205
    },
    {
      "epoch": 79,
      "train_loss": 0.19067314598295423,
      "validation_loss": 0.02781933732330799
    },
    {
      "epoch": 80,
      "train_loss": 0.214444556997882,
      "validation_loss": 0.029407747089862823
    },
    {
      "epoch": 81,
      "train_loss": 0.2170733114083608,
      "validation_loss": 0.028845131397247314
    },
    {
      "epoch": 82,
      "train_loss": 0.1486380613512463,
      "validation_loss": 0.027387768030166626
    },
    {
      "epoch": 83,
      "train_loss": 0.15032977031336892,
      "validation_loss": 0.027589203789830208
    },
    {
      "epoch": 84,
      "train_loss": 0.13614469021558762,
      "validation_loss": 0.02728942409157753
    },
    {
      "epoch": 85,
      "train_loss": 0.14345389273431566,
      "validation_loss": 0.027988625690340996
    },
    {
      "epoch": 86,
      "train_loss": 0.1219261387983958,
      "validation_loss": 0.028201399371027946
    },
    {
      "epoch": 87,
      "train_loss": 0.11477632903390461,
      "validation_loss": 0.027710700407624245
    },
    {
      "epoch": 88,
      "train_loss": 0.10778469012843238,
      "validation_loss": 0.028116729110479355
    },
    {
      "epoch": 89,
      "train_loss": 0.10677730747395092,
      "validation_loss": 0.028814328834414482
    },
    {
      "epoch": 90,
      "train_loss": 0.1050059124827385,
      "validation_loss": 0.02788354828953743
    },
    {
      "epoch": 91,
      "train_loss": 0.10887428911195861,
      "validation_loss": 0.028398742899298668
    },
    {
      "epoch": 92,
      "train_loss": 0.11731383452812831,
      "validation_loss": 0.02835109457373619
    },
    {
      "epoch": 93,
      "train_loss": 0.11323769100838238,
      "validation_loss": 0.027557555586099625
    },
    {
      "epoch": 94,
      "train_loss": 0.08863491771949662,
      "validation_loss": 0.028512826189398766
    },
    {
      "epoch": 95,
      "train_loss": 0.09047512461741765,
      "validation_loss": 0.02785918116569519
    },
    {
      "epoch": 96,
      "train_loss": 0.08597445901897219,
      "validation_loss": 0.027355879545211792
    },
    {
      "epoch": 97,
      "train_loss": 0.07618374832802349,
      "validation_loss": 0.0304588433355093
    },
    {
      "epoch": 98,
      "train_loss": 0.07836396619677544,
      "validation_loss": 0.02633563242852688
    },
    {
      "epoch": 99,
      "train_loss": 0.08456088602542877,
      "validation_loss": 0.03171861171722412
    },
    {
      "epoch": 100,
      "train_loss": 0.08044164834751023,
      "validation_loss": 0.02715528942644596
    },
    {
      "epoch": 101,
      "train_loss": 0.06947632423705524,
      "validation_loss": 0.027367105707526207
    },
    {
      "epoch": 102,
      "train_loss": 0.06838430340091388,
      "validation_loss": 0.028742333874106407
    },
    {
      "epoch": 103,
      "train_loss": 0.07111079494158427,
      "validation_loss": 0.027459466829895973
    },
    {
      "epoch": 104,
      "train_loss": 0.06613106197781032,
      "validation_loss": 0.026543425396084785
    },
    {
      "epoch": 105,
      "train_loss": 0.05674325654076205,
      "validation_loss": 0.02763577736914158
    },
    {
      "epoch": 106,
      "train_loss": 0.057295434176921844,
      "validation_loss": 0.02685701474547386
    },
    {
      "epoch": 107,
      "train_loss": 0.056492746704154544,
      "validation_loss": 0.027787763625383377
    },
    {
      "epoch": 108,
      "train_loss": 0.053525728483994804,
      "validation_loss": 0.026303201913833618
    },
    {
      "epoch": 109,
      "train_loss": 0.06478585261437628,
      "validation_loss": 0.026460867375135422
    },
    {
      "epoch": 110,
      "train_loss": 0.0605985207690133,
      "validation_loss": 0.026356639340519905
    },
    {
      "epoch": 111,
      "train_loss": 0.05344307091501024,
      "validation_loss": 0.030977880582213402
    },
    {
      "epoch": 112,
      "train_loss": 0.05181970964703295,
      "validation_loss": 0.026500534266233444
    },
    {
      "epoch": 113,
      "train_loss": 0.0476363365434938,
      "validation_loss": 0.026348577812314034
    },
    {
      "epoch": 114,
      "train_loss": 0.04603778074185053,
      "validation_loss": 0.026186509057879448
    },
    {
      "epoch": 115,
      "train_loss": 0.04945069468683667,
      "validation_loss": 0.027361024171113968
    },
    {
      "epoch": 116,
      "train_loss": 0.04474735860195425,
      "validation_loss": 0.026505853980779648
    },
    {
      "epoch": 117,
      "train_loss": 0.04405222895244757,
      "validation_loss": 0.027231652289628983
    },
    {
      "epoch": 118,
      "train_loss": 0.04525288608339098,
      "validation_loss": 0.02839997038245201
    },
    {
      "epoch": 119,
      "train_loss": 0.04431528515285916,
      "validation_loss": 0.02661093883216381
    },
    {
      "epoch": 120,
      "train_loss": 0.05040736786193318,
      "validation_loss": 0.02947883866727352
    },
    {
      "epoch": 121,
      "train_loss": 0.04758074341548814,
      "validation_loss": 0.025441475212574005
    },
    {
      "epoch": 122,
      "train_loss": 0.043562898205386266,
      "validation_loss": 0.028423162177205086
    },
    {
      "epoch": 123,
      "train_loss": 0.04139937957127889,
      "validation_loss": 0.026444651186466217
    },
    {
      "epoch": 124,
      "train_loss": 0.046702411853604846,
      "validation_loss": 0.029838791117072105
    },
    {
      "epoch": 125,
      "train_loss": 0.04529048647317621,
      "validation_loss": 0.026104379445314407
    },
    {
      "epoch": 126,
      "train_loss": 0.04580147316058477,
      "validation_loss": 0.026256779208779335
    },
    {
      "epoch": 127,
      "train_loss": 0.04018769868546062,
      "validation_loss": 0.024819640442728996
    },
    {
      "epoch": 128,
      "train_loss": 0.04398160345024533,
      "validation_loss": 0.027448197826743126
    },
    {
      "epoch": 129,
      "train_loss": 0.05022940432859792,
      "validation_loss": 0.024918368086218834
    },
    {
      "epoch": 130,
      "train_loss": 0.04246587637397978,
      "validation_loss": 0.03762327879667282
    },
    {
      "epoch": 131,
      "train_loss": 0.038160010137491755,
      "validation_loss": 0.026404498144984245
    },
    {
      "epoch": 132,
      "train_loss": 0.04024492659502559,
      "validation_loss": 0.027048174291849136
    },
    {
      "epoch": 133,
      "train_loss": 0.035712570779853396,
      "validation_loss": 0.028838779777288437
    },
    {
      "epoch": 134,
      "train_loss": 0.03514312683708138,
      "validation_loss": 0.02686319127678871
    },
    {
      "epoch": 135,
      "train_loss": 0.03535157752533754,
      "validation_loss": 0.02507440559566021
    },
    {
      "epoch": 136,
      "train_loss": 0.03744570600489775,
      "validation_loss": 0.02855067141354084
    },
    {
      "epoch": 137,
      "train_loss": 0.04168840994437536,
      "validation_loss": 0.029875541105866432
    },
    {
      "epoch": 138,
      "train_loss": 0.04529214836657047,
      "validation_loss": 0.030258595943450928
    },
    {
      "epoch": 139,
      "train_loss": 0.04095928039815691,
      "validation_loss": 0.03626580163836479
    },
    {
      "epoch": 140,
      "train_loss": 0.039235751868949995,
      "validation_loss": 0.026454966515302658
    },
    {
      "epoch": 141,
      "train_loss": 0.03294309808148278,
      "validation_loss": 0.02847306802868843
    },
    {
      "epoch": 142,
      "train_loss": 0.0316216213007768,
      "validation_loss": 0.027894549071788788
    },
    {
      "epoch": 143,
      "train_loss": 0.031084845256474283,
      "validation_loss": 0.02951812744140625
    },
    {
      "epoch": 144,
      "train_loss": 0.033452791472276054,
      "validation_loss": 0.02798672392964363
    },
    {
      "epoch": 145,
      "train_loss": 0.030347922092510596,
      "validation_loss": 0.02690262161195278
    },
    {
      "epoch": 146,
      "train_loss": 0.030686859869294696,
      "validation_loss": 0.030085371807217598
    },
    {
      "epoch": 147,
      "train_loss": 0.02938879591723283,
      "validation_loss": 0.030208511278033257
    },
    {
      "epoch": 148,
      "train_loss": 0.028088140508366957,
      "validation_loss": 0.029656410217285156
    },
    {
      "epoch": 149,
      "train_loss": 0.02875532965279288,
      "validation_loss": 0.026429763063788414
    },
    {
      "epoch": 150,
      "train_loss": 0.030761442250675626,
      "validation_loss": 0.026919538155198097
    },
    {
      "epoch": 151,
      "train_loss": 0.03037713695731428,
      "validation_loss": 0.02694716863334179
    },
    {
      "epoch": 152,
      "train_loss": 0.03272996284067631,
      "validation_loss": 0.028495170176029205
    },
    {
      "epoch": 153,
      "train_loss": 0.034361774308813944,
      "validation_loss": 0.027440598234534264
    },
    {
      "epoch": 154,
      "train_loss": 0.03148644810749425,
      "validation_loss": 0.029871447011828423
    },
    {
      "epoch": 155,
      "train_loss": 0.029023154742187925,
      "validation_loss": 0.026014668866991997
    },
    {
      "epoch": 156,
      "train_loss": 0.03048390481207106,
      "validation_loss": 0.02917322888970375
    },
    {
      "epoch": 157,
      "train_loss": 0.030146338666478794,
      "validation_loss": 0.030121490359306335
    },
    {
      "epoch": 158,
      "train_loss": 0.028631205049653847,
      "validation_loss": 0.028569376096129417
    },
    {
      "epoch": 159,
      "train_loss": 0.03604064819713434,
      "validation_loss": 0.026331476867198944
    },
    {
      "epoch": 160,
      "train_loss": 0.03351300830642382,
      "validation_loss": 0.024521654471755028
    },
    {
      "epoch": 161,
      "train_loss": 0.03120275752411948,
      "validation_loss": 0.028592895716428757
    },
    {
      "epoch": 162,
      "train_loss": 0.027639042379127607,
      "validation_loss": 0.029667368158698082
    },
    {
      "epoch": 163,
      "train_loss": 0.02839781033496062,
      "validation_loss": 0.030748095363378525
    },
    {
      "epoch": 164,
      "train_loss": 0.028761451029115252,
      "validation_loss": 0.02759188786149025
    },
    {
      "epoch": 165,
      "train_loss": 0.031054214677876897,
      "validation_loss": 0.03127138689160347
    },
    {
      "epoch": 166,
      "train_loss": 0.029346937934557598,
      "validation_loss": 0.03228887915611267
    },
    {
      "epoch": 167,
      "train_loss": 0.026825495581660006,
      "validation_loss": 0.03387800231575966
    },
    {
      "epoch": 168,
      "train_loss": 0.025938288826081488,
      "validation_loss": 0.02900281362235546
    },
    {
      "epoch": 169,
      "train_loss": 0.027091236060692206,
      "validation_loss": 0.028864901512861252
    },
    {
      "epoch": 170,
      "train_loss": 0.03139436348444886,
      "validation_loss": 0.02962927706539631
    },
    {
      "epoch": 171,
      "train_loss": 0.0359994876715872,
      "validation_loss": 0.026040105149149895
    },
    {
      "epoch": 172,
      "train_loss": 0.04502304933137364,
      "validation_loss": 0.024563977494835854
    },
    {
      "epoch": 173,
      "train_loss": 0.03657253997193442,
      "validation_loss": 0.025189857929944992
    },
    {
      "epoch": 174,
      "train_loss": 0.03242549258801672,
      "validation_loss": 0.03135109692811966
    },
    {
      "epoch": 175,
      "train_loss": 0.02717303733030955,
      "validation_loss": 0.030467664822936058
    },
    {
      "epoch": 176,
      "train_loss": 0.02508700845970048,
      "validation_loss": 0.029881060123443604
    },
    {
      "epoch": 177,
      "train_loss": 0.02606565712226762,
      "validation_loss": 0.03303920477628708
    },
    {
      "epoch": 178,
      "train_loss": 0.02497392541004552,
      "validation_loss": 0.032622355967760086
    },
    {
      "epoch": 179,
      "train_loss": 0.02368885734015041,
      "validation_loss": 0.02925308234989643
    },
    {
      "epoch": 180,
      "train_loss": 0.025272527916563883,
      "validation_loss": 0.02792762592434883
    },
    {
      "epoch": 181,
      "train_loss": 0.02493101203193267,
      "validation_loss": 0.031069304794073105
    },
    {
      "epoch": 182,
      "train_loss": 0.023012465486923855,
      "validation_loss": 0.03052770532667637
    },
    {
      "epoch": 183,
      "train_loss": 0.024436517101195123,
      "validation_loss": 0.03322775661945343
    },
    {
      "epoch": 184,
      "train_loss": 0.022599167707893584,
      "validation_loss": 0.02814207226037979
    },
    {
      "epoch": 185,
      "train_loss": 0.023190214505626097,
      "validation_loss": 0.037382327020168304
    },
    {
      "epoch": 186,
      "train_loss": 0.023432008922100067,
      "validation_loss": 0.03170586749911308
    },
    {
      "epoch": 187,
      "train_loss": 0.022896852654715378,
      "validation_loss": 0.03131787106394768
    },
    {
      "epoch": 188,
      "train_loss": 0.024250371381640434,
      "validation_loss": 0.029436279088258743
    },
    {
      "epoch": 189,
      "train_loss": 0.024204249183336895,
      "validation_loss": 0.026936590671539307
    },
    {
      "epoch": 190,
      "train_loss": 0.023643722240295675,
      "validation_loss": 0.038956645876169205
    },
    {
      "epoch": 191,
      "train_loss": 0.02256176434457302,
      "validation_loss": 0.03408857807517052
    },
    {
      "epoch": 192,
      "train_loss": 0.022148509716822043,
      "validation_loss": 0.0325651541352272
    },
    {
      "epoch": 193,
      "train_loss": 0.022070562984380458,
      "validation_loss": 0.03678849712014198
    },
    {
      "epoch": 194,
      "train_loss": 0.023581736410657566,
      "validation_loss": 0.02661200985312462
    },
    {
      "epoch": 195,
      "train_loss": 0.02242293239881595,
      "validation_loss": 0.040251508355140686
    },
    {
      "epoch": 196,
      "train_loss": 0.02098265342000458,
      "validation_loss": 0.02981674112379551
    },
    {
      "epoch": 197,
      "train_loss": 0.02067395754986339,
      "validation_loss": 0.031126365065574646
    },
    {
      "epoch": 198,
      "train_loss": 0.02100320543266005,
      "validation_loss": 0.031902581453323364
    },
    {
      "epoch": 199,
      "train_loss": 0.020748957991600037,
      "validation_loss": 0.03258749470114708
    },
    {
      "epoch": 200,
      "train_loss": 0.020801925617787573,
      "validation_loss": 0.029952960088849068
    },
    {
      "epoch": 201,
      "train_loss": 0.019912545879681904,
      "validation_loss": 0.03168858215212822
    },
    {
      "epoch": 202,
      "train_loss": 0.02007330550501744,
      "validation_loss": 0.03718625754117966
    },
    {
      "epoch": 203,
      "train_loss": 0.02068074471834633,
      "validation_loss": 0.03008192963898182
    },
    {
      "epoch": 204,
      "train_loss": 0.02051480973346366,
      "validation_loss": 0.03486629202961922
    },
    {
      "epoch": 205,
      "train_loss": 0.020167455698053043,
      "validation_loss": 0.028998445719480515
    },
    {
      "epoch": 206,
      "train_loss": 0.01980603403515286,
      "validation_loss": 0.03155353665351868
    },
    {
      "epoch": 207,
      "train_loss": 0.020126133432818785,
      "validation_loss": 0.029329124838113785
    },
    {
      "epoch": 208,
      "train_loss": 0.024211421195003722,
      "validation_loss": 0.02568674460053444
    },
    {
      "epoch": 209,
      "train_loss": 0.025865807404948607,
      "validation_loss": 0.030439142137765884
    },
    {
      "epoch": 210,
      "train_loss": 0.02525402760754029,
      "validation_loss": 0.02661645971238613
    },
    {
      "epoch": 211,
      "train_loss": 0.026985828143854935,
      "validation_loss": 0.028704583644866943
    },
    {
      "epoch": 212,
      "train_loss": 0.024306898315747578,
      "validation_loss": 0.02720213495194912
    },
    {
      "epoch": 213,
      "train_loss": 0.022549608618848853,
      "validation_loss": 0.034415103495121
    },
    {
      "epoch": 214,
      "train_loss": 0.02339524993052085,
      "validation_loss": 0.04703620821237564
    },
    {
      "epoch": 215,
      "train_loss": 0.021609879202312894,
      "validation_loss": 0.03458293154835701
    },
    {
      "epoch": 216,
      "train_loss": 0.02053349196083016,
      "validation_loss": 0.03395681083202362
    },
    {
      "epoch": 217,
      "train_loss": 0.020671469262904592,
      "validation_loss": 0.036784376949071884
    },
    {
      "epoch": 218,
      "train_loss": 0.01828942747993602,
      "validation_loss": 0.031720831990242004
    },
    {
      "epoch": 219,
      "train_loss": 0.0185004071229034,
      "validation_loss": 0.03304455429315567
    },
    {
      "epoch": 220,
      "train_loss": 0.01887317881402042,
      "validation_loss": 0.027607116848230362
    },
    {
      "epoch": 221,
      "train_loss": 0.019102932057446904,
      "validation_loss": 0.026148492470383644
    },
    {
      "epoch": 222,
      "train_loss": 0.019791159675353102,
      "validation_loss": 0.02895592898130417
    },
    {
      "epoch": 223,
      "train_loss": 0.020956682558688853,
      "validation_loss": 0.026906263083219528
    },
    {
      "epoch": 224,
      "train_loss": 0.019256837976475556,
      "validation_loss": 0.02820998802781105
    },
    {
      "epoch": 225,
      "train_loss": 0.018745291253758803,
      "validation_loss": 0.02950315736234188
    },
    {
      "epoch": 226,
      "train_loss": 0.018644776402248278,
      "validation_loss": 0.0359937883913517
    },
    {
      "epoch": 227,
      "train_loss": 0.018762296997010708,
      "validation_loss": 0.027200492098927498
    },
    {
      "epoch": 228,
      "train_loss": 0.01760865778972705,
      "validation_loss": 0.031649742275476456
    },
    {
      "epoch": 229,
      "train_loss": 0.016849760587016743,
      "validation_loss": 0.032226406037807465
    },
    {
      "epoch": 230,
      "train_loss": 0.016314004444413714,
      "validation_loss": 0.03082933835685253
    },
    {
      "epoch": 231,
      "train_loss": 0.016619682415492006,
      "validation_loss": 0.027413060888648033
    },
    {
      "epoch": 232,
      "train_loss": 0.017513493075966835,
      "validation_loss": 0.032062824815511703
    },
    {
      "epoch": 233,
      "train_loss": 0.0170768400033315,
      "validation_loss": 0.029749078676104546
    },
    {
      "epoch": 234,
      "train_loss": 0.01653682580217719,
      "validation_loss": 0.03468453139066696
    },
    {
      "epoch": 235,
      "train_loss": 0.01611387584772375,
      "validation_loss": 0.03229878470301628
    },
    {
      "epoch": 236,
      "train_loss": 0.016169019509106874,
      "validation_loss": 0.028849830850958824
    },
    {
      "epoch": 237,
      "train_loss": 0.016114740011592705,
      "validation_loss": 0.03132115676999092
    },
    {
      "epoch": 238,
      "train_loss": 0.015998271190457873,
      "validation_loss": 0.030036645010113716
    },
    {
      "epoch": 239,
      "train_loss": 0.016244068534837827,
      "validation_loss": 0.03132829815149307
    },
    {
      "epoch": 240,
      "train_loss": 0.01710547713769807,
      "validation_loss": 0.031975384801626205
    },
    {
      "epoch": 241,
      "train_loss": 0.015533358686500125,
      "validation_loss": 0.03126123547554016
    },
    {
      "epoch": 242,
      "train_loss": 0.016372835108389456,
      "validation_loss": 0.02927936613559723
    },
    {
      "epoch": 243,
      "train_loss": 0.015552237112489011,
      "validation_loss": 0.03385070711374283
    },
    {
      "epoch": 244,
      "train_loss": 0.01691955069286956,
      "validation_loss": 0.03485172986984253
    },
    {
      "epoch": 245,
      "train_loss": 0.019076902729769547,
      "validation_loss": 0.03365106135606766
    },
    {
      "epoch": 246,
      "train_loss": 0.01680615699539582,
      "validation_loss": 0.03857462853193283
    },
    {
      "epoch": 247,
      "train_loss": 0.015361751947138045,
      "validation_loss": 0.030608268454670906
    },
    {
      "epoch": 248,
      "train_loss": 0.015440694677333036,
      "validation_loss": 0.02607051096856594
    },
    {
      "epoch": 249,
      "train_loss": 0.017598660869730845,
      "validation_loss": 0.03020913526415825
    },
    {
      "epoch": 250,
      "train_loss": 0.016105781102346048,
      "validation_loss": 0.034502312541007996
    },
    {
      "epoch": 251,
      "train_loss": 0.016237017698585987,
      "validation_loss": 0.0397518165409565
    },
    {
      "epoch": 252,
      "train_loss": 0.016032443588806525,
      "validation_loss": 0.033637791872024536
    },
    {
      "epoch": 253,
      "train_loss": 0.014579487240148915,
      "validation_loss": 0.03214535862207413
    },
    {
      "epoch": 254,
      "train_loss": 0.014409963248504533,
      "validation_loss": 0.03701876848936081
    },
    {
      "epoch": 255,
      "train_loss": 0.014326559483177133,
      "validation_loss": 0.030674360692501068
    },
    {
      "epoch": 256,
      "train_loss": 0.015746612722675007,
      "validation_loss": 0.030833331868052483
    },
    {
      "epoch": 257,
      "train_loss": 0.015502805925077863,
      "validation_loss": 0.027103794738650322
    },
    {
      "epoch": 258,
      "train_loss": 0.014874172914359305,
      "validation_loss": 0.0277452040463686
    },
    {
      "epoch": 259,
      "train_loss": 0.014523968204028077,
      "validation_loss": 0.02864285185933113
    },
    {
      "epoch": 260,
      "train_loss": 0.0144692479322354,
      "validation_loss": 0.027164064347743988
    }
  ],
  "target_statistics_train_only": {
    "mean": 1.4755169650820337,
    "std": 24.121903621782927,
    "min": -113.74078369140625,
    "p1": -93.96059867858887,
    "p10": -3.8885838508605954,
    "median": -0.06153082847595215,
    "p90": 2.7742797851562497,
    "p99": 119.39946411132813,
    "max": 128.84457397460938
  },
  "target_normalization_mean": 1.4755169650820337,
  "target_normalization_std": 24.121903621782927,
  "split_metrics": {
    "train": {
      "action_sample_count": 1102,
      "ranking_group_count": 72,
      "mae": 3.055535464372892,
      "rmse": 4.680738101357094,
      "pearson": 0.7264792044506295,
      "spearman": 0.5551145825797781,
      "argmax_accuracy": 0.5138888888888888,
      "recall_at_1": 0.5138888888888888,
      "recall_at_3": 0.8472222222222222,
      "recall_at_5": 0.9027777777777778,
      "exact_regret": {
        "mean": 0.42717398837622667,
        "p95": 1.3131566047668461,
        "max": 15.63803956657648
      },
      "normalized_regret": {
        "mean": 0.056749988107546404,
        "p95": 0.3329374887246194,
        "max": 0.9061857417785375
      }
    },
    "validation": {
      "action_sample_count": 284,
      "ranking_group_count": 18,
      "mae": 1.8868212913165359,
      "rmse": 3.7773434038019427,
      "pearson": 0.21285042624803563,
      "spearman": 0.26675585931143136,
      "argmax_accuracy": 0.05555555555555555,
      "recall_at_1": 0.05555555555555555,
      "recall_at_3": 0.5555555555555556,
      "recall_at_5": 0.6111111111111112,
      "exact_regret": {
        "mean": 1.7420522736178503,
        "p95": 7.5618951112031905,
        "max": 9.303963363170624
      },
      "normalized_regret": {
        "mean": 0.5365505921168678,
        "p95": 0.9999999999942406,
        "max": 0.9999999999945418
      }
    },
    "held_out": {
      "action_sample_count": 536,
      "ranking_group_count": 36,
      "mae": 5.432069237443892,
      "rmse": 12.725422975709533,
      "pearson": 0.3210877224572545,
      "spearman": 0.40925591723562504,
      "argmax_accuracy": 0.3611111111111111,
      "recall_at_1": 0.3611111111111111,
      "recall_at_3": 0.6666666666666666,
      "recall_at_5": 0.8611111111111112,
      "exact_regret": {
        "mean": 5.583701190021303,
        "p95": 24.841391563415527,
        "max": 35.05346965789795
      },
      "normalized_regret": {
        "mean": 0.4095359751245398,
        "p95": 0.9999999999985596,
        "max": 0.9999999999999715
      }
    }
  },
  "train_seconds": 2.640425999998115,
  "evaluation_seconds": 0.0,
  "fit_status": "INCONCLUSIVE"
}
```

## G. Variant C — KEEP-relative two-tower

```json
null
```

## H. Root cause and recommendation

- Early stopping undertraining: `SUPPORTED`
- Target baseline mismatch: `SUPPORTED`
- Representation underfit: `INCONCLUSIVE`
- Root cause: `MIXED`
- Recommended target: `KEEP_RELATIVE_DELTA_Q`
- Recommended architecture: `CURRENT_MLP`
- These recommendations are diagnostic only; the production Cache Q network was not replaced.

## I. Runtime and execution guard

- Total offline diagnostic: `25.178843s`.
- Memorization probe: `19.869217s`.
- Variant A training: `2.564575s`.
- Variant B training: `2.640426s`.
- Variant C training: `0.000000s`.
- Additional environment / Matching / KKT executions: `0 / 0 / 0`.
