import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Poliklinik Öneri Chatbotu",
    page_icon="🏥",
    layout="centered"
)

# --- API ANAHTARI VE MODELLERİN YAPILANDIRILMASI ---
try:
    API_KEY = st.secrets["API_KEY"]
    genai.configure(api_key=API_KEY)
    embedding_model = 'models/text-embedding-004'
    generation_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"❌ HATA: API anahtarı yapılandırılamadı. Lütfen anahtarınızı kontrol edin. Detay: {e}")
    st.stop()

# --- VERİTABANI YÜKLEME VE KURULUMU (GÜÇLENDİRİLMİŞ) ---
@st.cache_resource
def setup_database():
    DB_PATH = "chroma_db"
    COLLECTION_NAME = "poliklinikler"
    
    # 1. CSV dosyasını en başta oku ve kontrol et
    try:
        df = pd.read_csv("semptom_veri_seti.csv")
        if df.empty:
            st.error("❌ 'semptom_veri_seti.csv' dosyası okundu ancak içi boş.")
            st.stop()
    except FileNotFoundError:
        st.error("❌ 'semptom_veri_seti.csv' dosyası bulunamadı. Lütfen GitHub reponuzda olduğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"❌ CSV dosyası okunurken hata: {e}")
        st.stop()

    # 2. Veritabanı klasörünü kontrol et
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        # Koleksiyonu almayı dene
        collection = client.get_collection(COLLECTION_NAME)
        doc_count = collection.count()
        
        # 3. Koleksiyon boş mu diye kontrol et
        if doc_count > 0:
            st.success(f"✅ Mevcut veritabanı başarıyla yüklendi ({doc_count} semptom).")
            return collection
        else:
            st.warning("Veritabanı bulundu ancak içi boş. Yeniden oluşturuluyor...")
            # Boş koleksiyonu sil
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)

    except Exception as e:
        st.warning(f"Veritabanı yüklenemedi veya bulunamadı: {e}. Yeni veritabanı oluşturuluyor...")
        # Hata varsa (örn: koleksiyon yoksa), oluştur
        try:
            collection = client.create_collection(name=COLLECTION_NAME)
        except chromadb.errors.UniqueConstraintError:
            # Nadir bir durum, koleksiyon var ama get_collection başarısız oldu
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)


    # 4. Veritabanını Doldur
    st.info("Veritabanı (sadece ilk çalıştırmada) hazırlanıyor... Bu 1-2 dakika sürebilir.")
    
    try:
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        progress_bar = st.progress(0, text="Embedding'ler oluşturuluyor...")

        for i, row in df.iterrows():
            semptom = row['semptom']
            poliklinik = row['poliklinik']
            
            if not isinstance(semptom, str) or not semptom.strip():
                continue

            try:
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=semptom,
                    task_type="RETRIEVAL_DOCUMENT"
                )['embedding']
                
                documents.append(semptom)
                embeddings.append(embedding)
                metadatas.append({'poliklinik': poliklinik})
                ids.append(str(i))
            except Exception as e:
                st.warning(f"'{semptom}' için embedding oluşturulamadı: {e}. Bu satır atlanıyor.")
            
            progress_bar.progress((i + 1) / len(df), text=f"Semptom {i+1}/{len(df)} işleniyor...")

        if not documents:
            st.error("❌ Veri setindeki semptomların hiçbiri için embedding oluşturulamadı.")
            st.stop()
            
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        progress_bar.empty()
        st.success(f"✅ Yeni veritabanı başarıyla oluşturuldu ve {len(documents)} semptom eklendi.")
        return collection

    except Exception as e:
        st.error(f"❌ HATA: Veritabanı doldurulurken bir sorun çıktı: {e}")
        st.stop()

# --- Veritabanını Yükle ---
try:
    collection = setup_database()
except Exception as e:
    st.error(f"❌ HATA: Veritabanı başlatılamadı. Detay: {e}")
    st.stop()


# --- FONKSİYONLAR ---
def en_yakin_poliklinigi_bul(soru, top_n=3):
    """Kullanıcının sorusunu embedding'e çevirir ve veritabanında en benzer N sonucu arar."""
    try:
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
    except Exception as e:
        st.error(f"Arama sırasında hata: {e}")
        return None

def cevap_uret(soru, bulunan_sonuclar):
    """Bulunan sonuçları ve kullanıcının sorusunu kullanarak Gemini ile bir cevap üretir."""
    
    if bulunan_sonuclar is None or not bulunan_sonuclar['documents'][0]:
        return "Üzgünüm, belirttiğiniz şikayetlerle ilgili bir poliklinik önerisi bulamadım. Lütfen bir sağlık kuruluşuna danışın."

    documents = bulunan_sonuclar['documents'][0]
    metadatas = bulunan_sonuclar['metadatas'][0]

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
    
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Cevap üretirken hata: {e}")
        return "Üzgünüm, cevabınızı işlerken bir sorunla karşılaştım."

# --- STREAMLIT ARAYÜZÜ (YENİ CHAT ARAYÜZÜ) ---
st.title("🏥 Poliklinik Öneri Chatbotu")
st.caption("Lütfen yaşadığınız sağlık sorununu veya belirtilerinizi aşağıdaki kutucuğa yazın.")

# Sohbet geçmişini hafızada tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni girdi al (st.text_input yerine st.chat_input)
if user_input := st.chat_input("Şikayetinizi buraya yazın..."):
    
    # Kullanıcının mesajını ekrana ve hafızaya ekle
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Chatbot'un cevabını oluştur
    with st.chat_message("assistant"):
        with st.spinner('Sizin için en uygun polikliniği arıyorum...'):
            benzer_sonuclar = en_yakin_poliklinigi_bul(user_input)
            chatbot_cevabi = cevap_uret(user_input, benzer_sonuclar)
            st.markdown(chatbot_cevabi)
    
    # Chatbot'un cevabını hafızaya ekle
    st.session_state.messages.append({"role": "assistant", "content": chatbot_cevabi})