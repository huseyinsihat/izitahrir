# İz-i Tahrir

Tahrir defteri sayfasını satırlara ayırıp okur.

## Çalıştırma

Windows’ta Python 3.11 yeter:

```powershell
.\baslat.cmd
```

Arayüz `http://localhost:7860` adresinde açılır. İlk çalıştırmada modeller iner.

```powershell
.\baslat.cmd -Api
```

API `http://localhost:8000` adresindedir. Belgeler `http://localhost:8000/docs` üzerinden yüklenir.

Ayrıntılı okuma için `.env` dosyasına `OPENROUTER_API_KEY` yazın. Örnek `.env.example` içindedir.
