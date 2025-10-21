import pandas as pd
import google.generativeai as genai
import chromadb
import os
from tqdm import tqdm 

print("Script başlatıldı...")

try:
    API_KEY = "LÜTFEN_KENDİ_API_ANAHTARINIZI_BURAYA_GİRİN"
    genai.configure(api_key=API_KEY)
    embedding_model = 'models/text-embedding-004'
    print("✅ API anahtarı ve model başarıyla yapılandırıldı.")
except Exception as e:
    print(f"❌ HATA: API anahtarı yapılandırılamadı. Detay: {e}")
    exit()

try:
   
    df = pd.read_csv("semptom_veri_seti.csv") 
    print(f"✅ 'semptom_veri_seti.csv' dosyası başarıyla okundu. {len(df)} satır veri bulundu.")
except FileNotFoundError:
    print("❌ HATA: 'semptom_veri_seti.csv' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()

try:
    client = chromadb.PersistentClient(path="chroma_db")
    
    collection = client.get_or_create_collection("poliklinikler") 
    print("✅ ChromaDB istemcisi başlatıldı ve 'poliklinikler' koleksiyonu hazır.")
except Exception as e:
    print(f"❌ HATA: ChromaDB başlatılamadı. Detay: {e}")
    exit()

print("\nVeriler işleniyor ve veritabanına ekleniyor. Bu işlem biraz zaman alabilir...")

documents = []
embeddings = []
metadatas = []
ids = []

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Embedding'ler oluşturuluyor"):
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
        ids.append(str(index)) 

    except Exception as e:
        print(f"⚠ UYARI: {index}. satırdaki '{semptom}' için embedding oluşturulamadı. Hata: {e}")

if documents:
    try:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"\n✅ Başarılı! {len(documents)} adet semptom veritabanına eklendi.")
    except Exception as e:
        print(f"❌ HATA: Veriler ChromaDB'ye eklenirken bir sorun oluştu. Detay: {e}")
else:
    print("⚠ UYARI: Veritabanına eklenecek geçerli bir veri bulunamadı.")

print("\nScript tamamlandı.")