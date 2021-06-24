import matplotlib.pyplot as plt

steps = [35008, 70016, 105024, 140000, 175008, 210016, 245024, 280000, 315008, 350016, 385024, 420000, 455008, 490016, 525024, 560023, 595031, 630007, 665015, 700023, 735031, 770007, 805015, 840023, 875031, 910007, 945015, 980023, 1015031, 1050007, 1085015, 1120014, 1155022, 1190030, 1225006, 1260014, 1295022, 1330030, 1365006, 1400014, 1435022, 1470030, 1505006, 1540014, 1575022, 1610030, 1645006, 1648101]
accuracies_8B = [0.7351148140621825, 0.7654948181263971, 0.7872383661857346, 0.7941475309896363, 0.799126193863036, 0.8057305425726479, 0.8098963625279415, 0.8250355618776671, 0.8207681365576103, 0.8294045925624873, 0.8271692745376956, 0.8292013818329608, 0.829302987197724, 0.8308270676691729, 0.8352977037187563, 0.8391587075797602, 0.8404795773216825, 0.8363137573663889, 0.8408859987807357, 0.8440357650883966, 0.8433245275350538, 0.8433245275350538, 0.8438325543588702, 0.849624060150376, 0.847693558219874, 0.8494208494208494, 0.8404795773216825, 0.8541963015647226, 0.8502336923389555, 0.8488112172322698, 0.8514529567161147, 0.8603942288152815, 0.8600894127209917, 0.858260516155253, 0.8605974395448079, 0.8587685429790693, 0.861003861003861, 0.8598862019914651, 0.8578540946962, 0.8596829912619386, 0.8615118878276773, 0.8599878073562284, 0.8647632595001016, 0.8613086770981507, 0.8635439951229424, 0.8624263361105466, 0.8631375736638894, 0.862934362934363]
losses_8B = [855.9996971487999, 690.6312810778618, 638.906450510025, 614.2332995086908, 599.2654203921556, 581.1839422732592, 563.279685869813, 557.2726286798716, 538.8863065093756, 539.4762713760138, 530.9508837610483, 519.4108180999756, 519.4013858735561, 515.0809799134731, 516.2402968853712, 485.923323944211, 454.41901184618473, 444.5391252115369, 449.6136126369238, 459.26869858801365, 446.3363569229841, 451.44738157093525, 458.6193386465311, 446.7868496850133, 447.5687161684036, 449.2585551291704, 451.41772462427616, 449.5418207794428, 441.69160944223404, 454.46753057837486, 441.88635266572237, 387.1921674236655, 353.86322160065174, 354.39714749902487, 351.38323249667883, 350.67424960434437, 348.0228600651026, 350.63183702528477, 350.44828563183546, 348.51546693220735, 353.2639208585024, 345.7732057198882, 352.3616674132645, 357.44835556298494, 354.3596162647009, 354.04249765723944, 355.3581337630749, 33.293133065104485]
accuracies_42B = [0.741211135947978, 0.7727087990245884, 0.7923186344238976, 0.7985165616744564, 0.8063401747612274, 0.8116236537289169, 0.8143669985775249, 0.8231050599471652, 0.8240195082300346, 0.8290997764681975, 0.8365169680959155, 0.8326559642349116, 0.8342816500711238, 0.839057102214997, 0.8405811826864459, 0.8440357650883966, 0.8414956309693152, 0.8395651290388132, 0.8453566348303191, 0.8470839260312945, 0.8482015850436903, 0.8453566348303191, 0.8446453972769762, 0.8525706157285105, 0.8511481406218249, 0.8532818532818532, 0.8437309489941069, 0.8560251981304613, 0.8506401137980085, 0.8499288762446657, 0.85318024791709, 0.8621215200162569, 0.8614102824629141, 0.8609022556390977, 0.8605974395448079, 0.8619183092867303, 0.8613086770981507, 0.8648648648648649, 0.8605974395448079, 0.8576508839666734, 0.8645600487705751, 0.8619183092867303, 0.8651696809591547, 0.8663889453363137, 0.8687258687258688, 0.8656777077829709, 0.8651696809591547, 0.8654744970534444]
losses_42B = [851.3273978531361, 673.6431069970131, 624.114491507411, 599.1627585440874, 585.0877138376236, 566.6675350219011, 549.0898998230696, 540.2368703335524, 530.0083706080914, 528.2885377854109, 519.4179846346378, 507.71051473915577, 509.73466551303864, 501.9226127117872, 502.32200433313847, 473.95688408613205, 442.55056734383106, 430.4489817172289, 440.5103995501995, 443.2793888002634, 433.3816327750683, 441.0043352395296, 445.5928122624755, 434.7799789458513, 430.6240329518914, 437.3851759135723, 438.7440169006586, 435.8896644860506, 431.50629702210426, 442.2033773511648, 429.6878005936742, 376.4562333598733, 341.02747935056686, 344.71584641188383, 338.8432426340878, 339.8452261351049, 335.5261914730072, 342.3384658396244, 333.40536350384355, 335.19074733555317, 339.85266552865505, 335.7461362667382, 340.04919695854187, 340.92805399000645, 341.73251209035516, 340.83251704648137, 341.1090506352484, 31.84779092669487]


plt.title(f'Dev Accuracy vs. seen examples')
plt.plot(steps, accuracies_8B)
plt.plot(steps, accuracies_42B)
plt.xlabel(f'seen examples')
plt.ylabel(f'Dev accuracy')
plt.ylim(0.7, 0.9)
plt.legend(['GloVe 8B', 'GloVe 42B'])
plt.show()


plt.title(f'Train loss vs. seen examples')
plt.plot(steps, losses_8B)
plt.plot(steps, losses_42B)
plt.xlabel(f'seen examples')
plt.ylabel(f'loss')
plt.legend(['GloVe 8B', 'GloVe 42B'])
plt.show()