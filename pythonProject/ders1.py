


#int tam sayıdır
#kelime = "merhaba" = string
#float = 0.5 = ondalık sayı

# print("hello world")

#print(15)

#print("merhaba")

#print(15.5)

#type sınıf öğrenmede işe yarar

#print(type(15.5))

#print((type(int(14.5))))

float = 15.4

convertedfloat = int(float)

#print(convertedfloat)

#print(15/5)
#print(15//5)
#print(15//7) #tam kısmını alıyor /tam sayı bölümü
#print(15%14) #sadece kalanı alır /kalan bulma
#print(3**2)  # **= üssü demektir

#index kaçıncı sırada ne var (0'DAN BAŞLAR)

yorumum = "ders akıcı gidiyor"

#print(yorumum[5]) # 0 dan başlaarak 5. terim
#print(yorumum[0 : 7]) # 0 dan başlayarak 7ye kadar alır (7 dahil değildir)
#print(yorumum[:8]) #baştan 8 e kadar
#print(yorumum[3:]) #3 ten sona kadar
#print(len(yorumum)) # ---- uzunluğu söyler

####################----------######################

#print("merhaba".isupper()) #str metodu büyük harfli mi ? islower küçük harfli mi AYRICA HEPSİ BÜYÜK YA DA KÜÇÜK OLMALI YOKSA FALSE


kelime1 = "naber"
kelime2 = "GELMEDİ SENDEN Bİ HABER"

#print(kelime1.upper()) #kelimeyi tak diye büyütür
#print(kelime2.lower()) #kelimeyi tak diye küçültür

###################------------################

adım = "benim adım şem"

#doğruadım = adım.replace("ş","c") # değiştirmek için kullanırız
#print(doğruadım)

##################-------------################

degisken = "benim;adım;cem"
degismis = degisken.split(";") # stringi ayırıp listeler ; ile ayrılanları yazıyor burda
#print(degismis)

sifre = " sifre2003 "

#print(sifre.strip())  #strip başta ve sondaki boşlukları siler
#print(sifre.lstrip()) #soldaki boşluk siler
#print(sifre.rstrip()) #sağdaki boşluk siler

#istenirse boşluk değil karakter girilebilir tabii

#.isnumeric parametresi sayı mı diye sormak oluyor
#.isalpha parametresi harf mi diye soruyor

#print("adım cen".startswith("adı")) # ..... yla mı başlıyor? ends with de var

##################--------------#############

#print("adım cem".count("m")) # kaç tane m var ?

sayı = int(input("sayı giriniz")) # giriş demek işte aq
if sayı<4:
    print("bu sayı 4 den küçük kanks")




