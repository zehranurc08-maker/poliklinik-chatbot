# 🏥 Poliklinik Öneri Chatbotu

Akbank GenAI Bootcamp kapsamında geliştirilen bu proje, kullanıcıların yazdığı sağlık şikayetlerine veya semptomlarına göre hangi polikliniğine gitmeleri gerektiği konusunda önerilerde bulunan, RAG (Retrieval-Augmented Generation) tabanlı bir chatbottur.

## 🚀 Web Arayüzü

Projeyi canlı olarak test etmek için aşağıdaki linki ziyaret edebilirsiniz:
➡ https://poliklinik-chatbot-b4qpah5tbdfhzyth3htmxn.streamlit.app/

## 🎯 Projenin Amacı

Sağlık sorunlarıyla karşılaşan birçok kişi, hangi tıbbi uzmana başvurması gerektiğini bilememektedir. Bu proje, bu bilgi eksikliğini gidermeyi amaçlamaktadır. Sağlık konusunda kullanıcıya doğrudan teşhis koymadan, sadece kullanıcıların doğal bir dille yazdığı şikayetlere göre uygun poliklinik yönlendirmesi yapan bir sistemdir.

## 📊 Veri Seti

Bu projede kullanılan veri seti, semptom_veri_seti.csv adında özel olarak oluşturulmuş bir dosyadır.

*Toplama Metodolojisi:* Veri seti, sık karşılaşılan sağlık sorunları ve bu sorunlarla ilgilenen temel tıp poliklinikleri üzerine yapılan araştırmalar sonucunda manuel olarak hazırlanmıştır.

*İçerik:* Veri seti üç ana sütundan oluşur:

* **semptom**: Sık rastlanan sağlık şikayetleri (örn: "Baş ağrısı, mide bulantısı").
* **poliklinik**: İlgili semptomla ilgilenen tıbbi bölüm (örn: "Nöroloji").
* **açıklama**: Semptom ve poliklinik arasındaki ilişkiyi açıklayan kısa bir not.

## 🛠 Kullanılan Teknolojiler ve Çözüm Mimarisi

Bu proje, bir RAG (Retrieval-Augmented Generation) mimarisi üzerine kurulmuştur. Akış aşağıdaki gibidir:

* *Web Arayüzü (Streamlit):* Kullanıcı girdisini almak ve chatbot cevabını göstermek için hızlı ve interaktif bir web arayüzü oluşturmada kullanılmıştır.
* **Generation Modeli (Google Gemini - gemini-2.5-flash):** Vektör veritabanından bulunan bilgilerle zenginleştirilmiş prompt'u işleyerek kullanıcıya nihai ve anlamlı bir cevap üretmek için kullanılmıştır.
* **Embedding Modeli (Google - models/text-embedding-004):** Hem veri setindeki semptomları hem de kullanıcının anlık sorusunu anlamsal olarak sayısal vektörlere dönüştürmek için kullanılmıştır.
* *Vektör Veritabanı (ChromaDB):* Oluşturulan embedding vektörlerini saklamak, yönetmek ve anlamsal arama (similarity search) yapmak için kullanılmıştır.
* *Veri İşleme (Pandas):* semptom_veri_seti.csv dosyasını okumak ve işlemek için veri_yukleyici.py adlı script dosyasının içinde kullanılmıştır.

### Çözüm Mimarisi (RAG Akışı)

Uygulamanın çalışma akışı aşağıdaki adımlardan oluşmaktadır:

1.  Kullanıcı, şikayetini Streamlit arayüzündeki metin kutusuna yazar.
2.  Girilen metin, Google'ın text-embedding-004 modeli kullanılarak anlamsal bir vektöre dönüştürülür.
3.  Bu vektör, ChromaDB veritabanına gönderilir. ChromaDB, sorgu vektörüne en yakın ve en ilgili semptom kayıtlarını (belgeleri) bulur.
4.  ChromaDB'den alınan ilgili semptomlar ve poliklinik bilgileri, kullanıcının orijinal sorusu ile birleştirilerek gemini-2.5-flash modeli için zenginleştirilmiş bir prompt (komut istemi) oluşturulur.
5.  Hazırlanan bu zenginleştirilmiş prompt, Gemini modeline gönderilir. Model, kendisine sağlanan bağlama dayanarak kullanıcıya tutarlı, doğru ve bağlama uygun bir poliklinik önerisi üretir.
6.  Üretilen cevap, kullanıcıya Streamlit arayüzünde gösterilir.

## ⚙ Kurulum ve Çalıştırma Kılavuzu

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

1.  *Projeyi Klonlayın:*
    bash
    git clone [https://github.com/](https://github.com/)[GITHUB KULLANICI ADINIZ]/[PROJE-REPO-ADINIZ].git
    cd [PROJE-REPO-ADINIZ]
    

2.  *Gerekli Kütüphaneleri Yükleyin:*
    Proje için gerekli tüm kütüphaneler requirements.txt dosyasında listelenmiştir.
    bash
    pip install -r requirements.txt
    

3.  *API Anahtarınızı Tanımlayın:*
    Projenin Google Gemini API'lerini kullanabilmesi için bir API anahtarına ihtiyacı vardır.
    * Proje ana dizininde .streamlit adında bir klasör oluşturun.
    * Bu klasörün içine secrets.toml adında bir dosya oluşturun.
    * Dosyanın içine kendi Google API anahtarınızı aşağıdaki gibi ekleyin:

    toml
    API_KEY = "AIzaSy...SİZİN-API-ANAHTARINIZ...L7JI"
    

4.  *Veritabanını Oluşturun:*
    Aşağıdaki komutu çalıştırarak semptom_veri_seti.csv dosyasındaki verilerin vektör veritabanını oluşturun. Bu komutun sadece bir kez çalıştırılması yeterlidir.
    bash
    python setup_database.py
    

5.  *Uygulamayı Başlatın:*
    Her şey hazır! Streamlit uygulamasını başlatmak için aşağıdaki komutu çalıştırın.
    bash
    streamlit run chatbot.py
    
    Uygulama, tarayıcınızda yerel bir adres üzerinden açılacaktır.

## 🖥 Web Arayüzü Kullanımı

Uygulama açıldığında sizi basit ve anlaşılır bir arayüz karşılayacaktır.

* *Giriş Alanı:* Ekranda bulunan metin kutusuna yaşadığınız sağlık sorununu veya semptomları yazın (Örn: "Son birkaç gündür şiddetli başım ağrıyor ve midem bulanıyor").
* *Cevap Bekleme:* Giriş yaptıktan sonra Enter'a basın. Chatbot, sizin için en uygun polikliniği ararken kısa bir bekleme animasyonu gösterecektir.
* *Öneriyi Okuma:* Chatbot, analizini tamamladıktan sonra size hangi polikliniğe gitmeniz gerektiği konusunda bir öneri sunacaktır.