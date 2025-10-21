import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
import os # Veritabanı klasörünün varlığını kontrol etmek için eklendi

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Poliklinik Öneri Chatbotu",
    page_icon="🏥",
    layout="centered"
)

# --- API ANAHTARI VE MODELLERİN YAPILANDIRILMASI ---
try:
    # Streamlit Cloud'un "Secrets" bölümünden API anahtarını al
    API_KEY = st.secrets["API_KEY"]
    
    genai.configure(api_key=API_KEY)
    embedding_model = 'models/text-embedding-004'
    generation_model = genai.GenerativeModel('gemini-1.5-flash') # 1.5-flash daha yaygın, 2.5-flash yerine bunu kullanabiliriz
except Exception as e:
    st.error(f"❌ HATA: API anahtarı yapılandırılamadı. Lütfen anahtarınızı kontrol edin. Detay: {e}")
    st.stop()

# --- VERİTABANI YÜKLEME VE KURULUMU (CACHE İLE) ---
# Bu @st.cache_resource, fonksiyonun sadece bir kez (ilk çalıştırmada) çalışmasını sağlar,
# böylece her kullanıcı sorgusunda veritabanını yeniden oluşturmaz.
@st.cache_resource
def setup_database():
    DB_PATH = "chroma_db"
    
    # 1. Veritabanı zaten var mı diye kontrol et
    if os.path.exists(DB_PATH):
        st.info("Mevcut veritabanı yükleniyor...")
        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_collection("poliklinikler")
            st.success("✅ Veritabanı başarıyla yüklendi.")
            return collection
        except Exception as e:
            st.warning(f"Veritabanı bulundu ancak yüklenemedi: {e}. Yeniden oluşturulacak.")
            # Eğer klasör var ama içi bozuksa, yeniden oluşturmaya izin ver

    # 2. Veritabanı yoksa veya bozuksa, oluştur
    st.warning("Veritabanı bulunamadı. Yeni veritabanı oluşturuluyor...")
    st.info("Bu işlem (sadece ilk çalıştırmada) birkaç dakika sürebilir.")
    
    try:
        # Veri setini oku
        df = pd.read_csv("semptom_veri_seti.csv")
        
        # Chroma client'ı başlat
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection("poliklinikler")

        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        # Streamlit'te ilerleme çubuğu göster
        progress_bar = st.progress(0, text="Embedding'ler oluşturuluyor...")

        for i, row in df.iterrows():
            semptom = row['semptom']
            poliklinik = row['poliklinik']
            
            if not isinstance(semptom, str) or not semptom.strip():
                continue

            embedding = genai.embed_content(
                model=embedding_model,
                content=semptom,
                task_type="RETRIEVAL_DOCUMENT"
            )['embedding']
            
            documents.append(semptom)
            embeddings.append(embedding)
            metadatas.append({'poliklinik': poliklinik})
            ids.append(str(i))
            
            # İlerleme çubuğunu güncelle
            progress_bar.progress((i + 1) / len(df), text=f"Semptom {i+1}/{len(df)} işleniyor...")

        # Toplu ekleme
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        progress_bar.empty() # İlerleme çubuğunu kaldır
        st.success(f"✅ Yeni veritabanı başarıyla oluşturuldu ve {len(documents)} semptom eklendi.")
        return collection

    except Exception as e:
        st.error(f"❌ HATA: Veritabanı oluşturulurken bir sorun çıktı: {e}")
        st.stop()

# --- CHROMA VERİTABANINA BAĞLANMA ---
# (Eski bağlantı kodunun yerini bu yeni fonksiyon alıyor)
try:
    # Veritabanını yükle veya oluştur
    collection = setup_database() 
except Exception as e:
    st.error(f"❌ HATA: Veritabanı başlatılamadı. Detay: {e}")
    st.stop()

# --- FONKSİYONLAR ---
def en_yakin_poliklinigi_bul(soru, top_n=3):
    """Kullanıcının sorusunu embedding'e çevirir ve veritabanında en benzer N sonucu arar."""
    soru_embedding = genai.embed_content(
        model=embedding_model,
        content=soru,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    
    results = collection.query(
        query_embeddings=[soru_embedding],
        n_results=top_n
    )
    return results

def cevap_uret(soru, bulunan_sonuclar):
    """Bulunan sonuçları ve kullanıcının sorusunu kullanarak Gemini ile bir cevap üretir."""
    documents = bulunan_sonuclar['documents'][0]
    metadatas = bulunan_sonuclar['metadatas'][0]
    
    if not documents:
        return "Üzgünüm, belirttiğiniz şikayetlerle ilgili bir poliklinik önerisi bulamadım. Lütfen bir sağlık kuruluşuna danışın."

    context = "\n".join([f"İlgili Semptom: {doc}, Gitmesi Gereken Poliklinik: {meta['poliklinik']}" for doc, meta in zip(documents, metadatas)])

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
    
    response = generation_model.generate_content(prompt)
    return response.text

# --- STREAMLIT ARAYÜZÜ ---
st.title("🏥 Poliklinik Öneri Chatbotu")
st.caption("Lütfen yaşadığınız sağlık sorununu veya belirtilerinizi aşağıdaki kutucuğa yazın.")

# Kullanıcıdan metin girdisi al
user_input = st.text_input("Şikayetinizi buraya yazın...", key="user_input")

if user_input:
    with st.spinner('Sizin için en uygun polikliniği arıyorum... Lütfen bekleyin...'):
        # 1. Veritabanında arama yap
        benzer_sonuclar = en_yakin_poliklinigi_bul(user_input)
        
        # 2. Bulunan sonuçlarla bir cevap üret
        chatbot_cevabi = cevap_uret(user_input, benzer_sonuclar)
        
        # 3. Sonucu ekrana yazdır
        st.markdown("---")
        st.write("🤖 *Chatbot'un Önerisi:*")
        st.info(chatbot_cevabi)