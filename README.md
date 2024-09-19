# NISA DAGASAN MACHINE LEARNING BOOTCAMP

 ## KAGGLE LINK = https://www.kaggle.com/code/nisadaaan/notebookbf04dbb841

# Proje: Kredi Kartı Sahtekarlığı Tespiti
**Proje Tanıtımı**
Aygaz Makine Öğrenmesi Bootcamp: Yeni Nesil Proje Kampı kapsamında gerçekleştirilen bu projede, katılımcılar makine öğrenmesi alanında Gözetimli ve Gözetimsiz Öğrenme teknikleri ile çalışacaklardır. Bu tekniklerle veriler, kategoriler halinde sınıflandırılacak, sürekli değerler tahmin edilecek ya da girdilere dayalı olarak kümelendirilecektir. Proje, yapay zeka ve makine öğrenmesi alanında veri analizi, model geliştirme ve değerlendirme konularında katılımcılara pratik deneyim kazandırmayı amaçlamaktadır. Proje Kaggle platformunda yürütülecek olup, katılımcılar seçecekleri veri setinde hem Gözetimli hem de Gözetimsiz Öğrenme yöntemlerini uygulayacaklardır.

Bu proje, kredi kartı işlemleri sırasında gerçekleşen sahtekarlıkları tespit etmeyi amaçlayan bir makine öğrenmesi çalışmasıdır. Gözetimli ve gözetimsiz öğrenme teknikleri kullanılarak verilerin analizi, model geliştirme ve değerlendirme süreçleri gerçekleştirilecektir. Bu sayede, kullanıcıların haberi olmadan yapılan sahtekarlık işlemleri tespit edilerek, maddi kayıpların önüne geçilmesi hedeflenmektedir.

**Veri Kümesi**
Kullanılan veri kümesi, Avrupa'daki kart sahipleri tarafından Eylül 2013'te yapılan kredi kartı işlemlerini içermektedir. Veri kümesi iki gün boyunca yapılan 284,807 işlemden oluşmaktadır ve bunların 492'si sahte işlem olarak etiketlenmiştir. Bu nedenle veri oldukça dengesizdir; sahtekarlık işlemleri toplam işlemlerin yalnızca %0.172'sini oluşturmaktadır.

Veri setindeki tüm giriş değişkenleri, gizlilik nedenleriyle PCA (Principal Component Analysis) dönüşümü ile oluşturulmuştur. 'Time' ve 'Amount' özellikleri dışındaki tüm özellikler PCA ile elde edilmiştir.

Time: Her bir işlemin, veri kümesindeki ilk işlemden sonra geçen saniye cinsinden süresini gösterir.
Amount: İşlemin tutarını belirtir ve maliyete duyarlı öğrenmede kullanılabilir.
Class: Hedef değişken olup, 1 değeri sahte işlemleri, 0 ise normal işlemleri temsil eder.
Sınıflandırma Zorlukları
Veri setindeki sınıfların dengesizliği nedeniyle, doğruluk ölçümü için Kesinlik-Recall Eğrisi Altındaki Alan (AUPRC) kullanılması önerilmektedir. Klasik karışıklık matrisi doğruluğu, dengesiz sınıflar için yanıltıcı sonuçlar verebilir.


# Gerekli Kütüphaneler

Bu projede aşağıdaki kütüphaneler kullanılmaktadır:

- `sklearn.cluster.DBSCAN`: DBSCAN algoritması için.
- `sklearn.metrics.silhouette_score`: Kümeleme performansını ölçmek için Silhouette Skoru.
- `sklearn.model_selection.ParameterGrid`: Grid arama için manuel parametre kombinasyonları oluşturur.
- `sklearn.preprocessing.StandardScaler`: Verileri standartlaştırmak için.
- `sklearn.decomposition.PCA`: Boyut azaltma işlemleri için.
- `sklearn.cluster.KMeans`: KMeans algoritması için.
- `sklearn.pipeline.make_pipeline`: Pipeline oluşturmak için.
- `scipy.cluster.linkage`: Hiyerarşik kümeleme için bağlantı matrisini oluşturur.
- `scipy.cluster.dendrogram`: Dendrogram görselleştirmesi için.
- `seaborn`: Gelişmiş görselleştirmeler için.
- `warnings`: Uyarıları yönetmek için.
- `sklearn.impute.SimpleImputer`: Eksik verileri doldurmak için.
- `sklearn.model_selection.train_test_split`: Verileri eğitim ve test setlerine ayırmak için.
- `sklearn.linear_model.LogisticRegression`: Lojistik regresyon modeli için.
- `sklearn.svm.SVC`: Destek vektör makinesi sınıflandırıcısı için.
- `sklearn.tree.DecisionTreeClassifier`: Karar ağaçları için.
- `sklearn.metrics`: Sınıflandırma performansını ölçmek için (doğruluk, kesinlik, hatırlama, F1 skoru, sınıflandırma raporu).

Bu kütüphaneleri kullanarak projenizde gerekli analizleri ve modellemeleri gerçekleştirebilirsiniz.

2. EDA (Exploratory Data Analysis)
2.1 Genel Bilgiler
EDA, veri setimizi daha iyi anlamak ve modelleme için gereken öngörüleri elde etmek amacıyla yapılan bir süreçtir. Veriyi incelemek için kullanılan temel fonksiyonlar:

df.info(): Veri setinin yapısı hakkında genel bilgi verir. Özelliklerin isimleri, kaç satır ve sütun içerdiği gibi bilgiler sunar.

Örnek Çıktı: 284807 satır ve 31 sütun içeriyor.
df.describe(): Sayısal sütunların temel istatistiklerini sağlar. Her bir değişken için sayı, ortalama, minimum ve maksimum değerler gibi özet bilgiler sunar.

df.head(): Veri setinin ilk 5 satırını göstererek veri yapısını gözlemlememizi sağlar. Tüm sütunların sayısal değerler içerdiği tespit edilmiştir.

2.2 Sınıf Dağılımı
Veri setimizde dolandırıcılık (fraud) ve dolandırıcılık olmayan (non-fraud) işlem sayıları arasındaki dağılımı analiz etmek için bir countplot oluşturduk. İşlemler şu şekilde gerçekleştirilmiştir:

Veri Hazırlığı: Class sütunu, dolandırıcılık durumunu ifade etmektedir. Bu sütun, "Fraud Status" olarak yeniden adlandırıldı. Burada 0 değeri non-fraud, 1 değeri ise fraud işlemlerini temsil etmektedir.

Görselleştirme: Seaborn kütüphanesinin countplot fonksiyonu ile Fraud Status sütununa göre veri dağılımını çizdik.

Sonuç: Grafikte, non-fraud işlemlerin sayısının oldukça fazla olduğunu, fraud işlemlerin ise çok az olduğunu gözlemledik. Non-Fraud işlemlerin sayısı yaklaşık 270.000 civarındayken, Fraud işlemlerin sayısı neredeyse yok denecek kadar azdır. Bu durum, veri setindeki dengesiz dağılımı gözler önüne sermektedir.
2.3 Standardizasyon
Veri setindeki Amount sütunu standardize edilmiştir. Standardizasyon, verilerin ortalama değerinin 0, standart sapmasının ise 1 olmasını hedefler. Bu işlem, farklı ölçeklerdeki verilerin daha anlamlı hale gelmesini sağlar:

python
Copy code
df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))
İlk Görüntüleme: İlk birkaç satırı görüntülemek için print(df['Amount'].head()) komutunu kullandık.

İstatistikler: Ortalama (mean) ve standart sapma (std) hesaplamaları yapıldı. Ortalama yaklaşık sıfır, standart sapma ise yaklaşık bir çıkmalıdır.

Sonuç:

Ortalama: 2.913951958230651e-17 (yaklaşık sıfır)
Standart Sapma: 1.000001755579451 (yaklaşık bir)
2.4 Görselleştirme
Log-Transformed Amount Distribution Plot oluşturarak Count vs Log(Amount+1) grafiği çizildi. Bu grafikte, Amount sütunundaki veriler üzerinde logaritmik bir dönüşüm uygulanmıştır. Sonuç olarak:

Düşük miktarlarda belirgin bir yoğunluk görülmektedir. Yüksek miktarlarda ise işlem sayısının azaldığı gözlemlenmiştir.

Sonuç: Bu tür bir dağılım, finansal işlemlerde yaygın olarak görülen bir durumdur ve logaritmik dönüşüm kullanarak küçük ve büyük işlemler arasındaki fark daha net bir şekilde görselleştirilmiştir.

2.5 Korelasyon Matrisi
Fraud Status (Class) sütunu ile diğer değişkenler arasındaki korelasyonları inceledik. Korelasyon matrisi, değişkenlerin birbirleriyle olan ilişkilerini anlamak için önemli bir araçtır. Öne çıkan noktalar:

LogAmount değişkeni ile Fraud Status arasında düşük bir korelasyon bulunmaktadır. Bu durum, işlemin miktarının dolandırıcılığı belirlemede çok etkili olmadığını göstermektedir.

V değişkenleri arasında zayıf ilişkiler gözlemlenmiştir. Özellikle V1 ile V28 arasında güçlü bir korelasyon yoktur.

3. Model Eğitimi ve Parametre Seçimi
3.1 Lojistik Regresyon
Lojistik regresyon modeli ile dolandırıcılık tespiti için aşağıdaki adımlar izlenmiştir:

Parametre Ayarlaması: C değeri üzerinde arama yapıldı. C, modelin karmaşıklığını dengeleyen bir düzenleme parametresidir.
Kullanılan Parametreler: C: [0.001, 0.01, 0.1, 1, 10, 100]
Sonuç:
En İyi Parametreler: {'C': 1}
En İyi Skor: 0.9456
3.2 Destek Vektör Makineleri (SVM)
SVM modeli için ise aşağıdaki adımlar izlenmiştir:

Kernel Seçenekleri: RBF ve linear kernel seçenekleri değerlendirildi.

Kullanılan Parametreler:
C: [0.1, 1, 10, 100]
gamma: [1, 0.1, 0.01, 0.001]
kernel: ['rbf', 'linear']
Sonuç:

En İyi Parametreler: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
En İyi Skor: 0.9543
3.3 Model Değerlendirmeleri
Her iki modelin performans metrikleri aşağıdaki gibidir:

Doğruluk (Accuracy):
Lojistik Regresyon: 0.9991
SVM: 0.9993
Kesinlik (Precision):
Lojistik Regresyon: 0.8636
SVM: 0.9683
Duyarlılık (Recall):
Lojistik Regresyon: 0.5816
SVM: 0.6224
F1 Skoru:
Lojistik Regresyon: 0.6951
SVM: 0.7578
4. Unsurpervised Learning
4.1 K-Means ve DBSCAN
K-Means algoritması ile kümeleme yapıldı ve silhouette skoru değerlendirildi:

Silhouette Skoru:
KMeans: 0.4012
DBSCAN: 0.3124
Hangi algoritma daha iyi?
Sonuçlar, KMeans algoritmasının bu veri seti için daha iyi performans gösterdiğini ortaya koymaktadır. KMeans, daha yüksek bir silhouette skoru ile kümeler arasındaki ayrımı daha iyi yapmıştır. Özellikle, veri noktalarının kümeler arasında daha belirgin sınırlarla ayrılması gereken durumlarda KMeans, DBSCAN'a göre daha etkili olabilmektedir. Bununla birlikte, DBSCAN, düzensiz şekilli kümeleri veya gürültülü verileri ayıklamada daha iyi performans gösterme potansiyeline sahiptir. Bu veri setinde, veriler KMeans için daha uygun bir yapıya sahip görünmektedir




