import streamlit as st  
import pandas as pd  
import google.generativeai as genai  
import chromadb  
import os  

# --- SAYFA YAPILANDIRMASI ---

st.set_page_config(
    page_title="Poliklinik Öneri Chatbotu",  # Tarayıcı sekmesinde görünecek başlık
    page_icon="🏥",                          # Tarayıcı sekmesinde görünecek ikon 
    layout="centered"                        # İçeriğin sayfanın ortasında hizalanmasını sağlar
)

# --- API ANAHTARI VE MODELLERİN YAPILANDIRILMASI ---
try:
     # Streamlit'in "Secrets" özelliğini kullanarak API anahtarını güvenli bir şekilde sakladık
    # Proje klasöründe .streamlit/secrets.toml dosyası oluşturup içine API_KEY = "..." yazmalısınız.
    # Bu yöntem, API anahtarınızın kod içinde görünmesini ve paylaşılmasını engeller.
    API_KEY = st.secrets["API_KEY"]
    
    # Google AI servisini, aldığımız API anahtarı ile yapılandırırız.
    genai.configure(api_key=API_KEY)
    
    # Embedding için kullanılacak modelin adını belirliyoruz.
    
    embedding_model = 'models/text-embedding-004'
    
    # Cevap üretmek için kullanılacak üretken YZ modelini (Gemini Flash) başlatıyoruz.
    generation_model = genai.GenerativeModel('gemini-2.5-flash')

except Exception as e:
    # Eğer try bloğunda (örn. API anahtarı bulunamazsa) bir hata oluşursa:
    # Ekrana bir hata mesajı basarız.
    st.error(f"❌ HATA: API anahtarı yapılandırılamadı. Lütfen anahtarınızı kontrol edin. Detay: {e}")
    # st.stop() komutu, programın geri kalanının çalışmasını durdurur.
    st.stop()

# --- CHROMA VERİTABANINA BAĞLANMA ---
try:
    # 'chroma_db' klasöründe bulunan kalıcı veritabanına bağlanıyoruz.
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Veritabanı içindeki 'poliklinikler' adlı koleksiyonu çekiyoruz.
    # Tüm verilerimiz bu koleksiyon içinde saklanır.
    collection = client.get_collection("poliklinikler")
    
except Exception as e:
    # Eğer veritabanı bağlantısında (örn. 'chroma_db' klasörü yoksa) bir hata olursa:
    st.error(f"❌ HATA: ChromaDB veritabanına bağlanılamadı. 'chroma_db' klasörünün olduğundan emin olun.")
    st.info("💡 İpucu: Veritabanını oluşturmak için bir önceki kod versiyonunu (veritabanı oluşturma kodu) bir kez çalıştırmanız gerekebilir.")
    st.stop() 

# --- FONKSİYONLAR ---

def en_yakin_poliklinigi_bul(soru, top_n=3):
    """Kullanıcının sorusunu embedding'e çevirir ve veritabanında en benzer N sonucu arar."""
    
    # 1. Kullanıcının sorusunu bir embedding'e dönüştür.
    soru_embedding = genai.embed_content(
        model=embedding_model,          # Hangi embedding modelini kullanacağımız
        content=soru,                   # Hangi metni vektöre çevireceğimiz (kullanıcının sorusu)
        task_type="RETRIEVAL_QUERY"     # Bu embedding'in bir arama sorgusu için olduğunu belirtir (daha iyi sonuçlar için optimizasyon sağlar)
    )['embedding']                      # Dönen sonucun içinden sadece 'embedding' listesini alır.
    
    # 2. ChromaDB koleksiyonu içinde bu vektöre en çok benzeyen sonuçları ara.
    results = collection.query(
        query_embeddings=[soru_embedding],  # Arama yapmak için kullanıcının soru vektörü
        n_results=top_n                     # En benzer kaç sonucu getireceğini belirtir (varsayılan 3)
    )
    
    # 3. Bulunan sonuçları (dokümanlar, metadatalar, mesafeler) geri döndür.
    return results

def cevap_uret(soru, bulunan_sonuclar):
    """Bulunan sonuçları (context) ve kullanıcının sorusunu kullanarak Gemini ile bir cevap üretir."""
    
    # ChromaDB'den dönen sonuçları ayıklar.
    # 'documents' aranan metne karşılık gelen semptom metinleridir.
    documents = bulunan_sonuclar['documents'][0]
    # 'metadatas' o semptomlara bağlı poliklinik bilgileridir.
    metadatas = bulunan_sonuclar['metadatas'][0]
    
    # Eğer 'documents' listesi boşsa, yani veritabanında hiçbir benzer sonuç bulunamamışsa:
    if not documents:
        # Standart bir "bulunamadı" mesajı döndürür.
        return "Üzgünüm, belirttiğiniz şikayetlerle ilgili bir poliklinik önerisi bulamadım. Lütfen bir sağlık kuruluşuna danışın."

    # Bulunan sonuçları (ilgili semptom ve poliklinik) modelin anlayacağı bir 'context' metnine dönüştürürüz.
    # zip(documents, metadatas) -> (semptom1, poliklinik1), (semptom2, poliklinik2)... eşleştirmesi yapar.
    context = "\n".join([f"İlgili Semptom: {doc}, Gitmesi Gereken Poliklinik: {meta['poliklinik']}" for doc, meta in zip(documents, metadatas)])

    # Gemini modeline göndereceğimiz prompt hazırlıyoruz.
    # Bu, modelin nasıl davranacağını, hangi bilgileri kullanacağını belirten bir şablondur.
    prompt = f"""
    Sen, kullanıcının sağlık sorunlarına göre hangi polikliniğe gitmesi gerektiğini öneren bir yardımcı asistansın.
    Sana kullanıcının sorusunu ve veritabanından bulduğun en ilgili semptomları vereceğim.
    Bu bilgilere dayanarak, kullanıcıya nazik, samimi ve anlaşılır bir dille öneride bulun.
    Cevabında tıbbi bir teşhis koymadığını, sadece bir yönlendirme yaptığını belirt.
    Sadece verdiğim bilgi metnindeki poliklinikleri öner.
    
    VERİLEN BİLGİ METNİ:
    {context}
    
    KULLANICININ SORUSU:
    "{soru}"
    
    CEVAP:
    """
    
    # Hazırlanan prompt'u Gemini modeline göndeririz.
    response = generation_model.generate_content(prompt)
    
    # Modelden gelen cevabın sadece metin kısmını geri döndürürüz.
    return response.text

# --- STREAMLIT ARAYÜZÜ (Uygulamanın ana akışı) ---


st.title("🏥 Poliklinik Öneri Chatbotu")
st.caption("Lütfen yaşadığınız sağlık sorununu veya belirtilerinizi aşağıdaki kutucuğa yazın.")

# Girilen metin 'user_input' değişkenine atanır.
user_input = st.text_input("Şikayetinizi buraya yazın...", key="user_input")

# 'if user_input:' bloğu, kullanıcı kutucuğa bir şey yazıp Enter'a bastığı anda çalışır.
if user_input:
    # 'with st.spinner(...)' bloğu, içindeki kodlar çalışırken kullanıcıya bir bekleme mesajı gösterir.
    with st.spinner('Sizin için en uygun polikliniği arıyorum... Lütfen bekleyin...'):
        
        # 1. Adım: Kullanıcının girdiği metni (user_input) al ve veritabanında en benzer sonuçları bul.
        benzer_sonuclar = en_yakin_poliklinigi_bul(user_input)
        
        # 2. Adım: Bulunan benzer sonuçları (benzer_sonuclar) ve kullanıcının sorusunu (user_input)
        #           'cevap_uret' fonksiyonuna göndererek modelden bir cevap oluşturmasını iste.
        chatbot_cevabi = cevap_uret(user_input, benzer_sonuclar)
        
        # 3. Adım: Üretilen cevabı kullanıcıya göster.
        st.markdown("---") 
        st.write("🤖 Chatbot'un Önerisi:") 
        st.info(chatbot_cevabi)