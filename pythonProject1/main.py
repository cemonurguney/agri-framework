a = "merhaba dünya"
b = "güle güle dünya"
print(a + " " + b);  # \n satır demek  \t tab demek 3 tırnak da satır oluyor.

print(a[5])  # 5. eleman (0 dan başlayarak) -1 de en sondur
print(a[0:5])  # 1. dahil 2.(burda 5. eleman) dahil değil arasını yazdır demek
print(a[::-1]) # boşluklar baş ve sonu 3. yazılan kaçar kaçar olduğu, -1 yazarsak tersten yazar
print(a.upper())
print(a.capitalize()) #baş harfi büyütür
print(a.startswith("m"))# böyle bailayıp başlamadığını soruyor- endswith hali de var
print(len(a)) # uzunluk sormaca (0dan başlar unutma
print(a*10)#10 defa yazar
print("{} {} beraberdir.".format(a,b))#küme gösterip değişkeni oraya bağlamak
print(f"{a} {b} beraberdir.") # yukardakinin aynısı parantez içlisi