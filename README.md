# PedestrianCounter
## Krótko o aplikacji
Celem aplikacji jest zliczanie przechodniów.
Istnieje wiele sposób na rozwiązanie tego problemu jak chociażby detekcja ludzi za pomocą wbudowanego w OpenCV HOG + SVM, żeby zmniejszyć zasobożerność można to połaczyć z trackerami obiektów jak np. MOSSE. 
Jednak w moim przypadku zastosowałem metody oddzielania tła, konkretnie użyłem mieszanin gaussowskich(MOG).
W dalszej części przeprowadziłem postprocessing, użyłem transformacji morfologicznych takich jak zamykanie, otwieranie, dylatacja i progowanie.
W celu uzyskania centroidów obiektów użyłem momentów. Z tej pozycji wystarczył dość prosty algorytm do zliczania przechodniów. 
