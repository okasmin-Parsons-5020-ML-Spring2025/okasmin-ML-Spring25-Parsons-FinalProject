api documentation: https://metmuseum.github.io/

example call last objects: https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=%22vessel%22

example call single object: https://collectionapi.metmuseum.org/public/collection/v1/objects/479496

---

js
[x] function to download image given object id - use primaryImageSmall field

python
[x] make images bw --> may not need this for models, didn't save new image files
[x] explore BRIA model --> success
[x] explore BiRefNet model --> failed
[x] save copy of image with background removed --> did this from BRIA model

---

js
[] download 1000 images
[] save data for 1000 objects to df / json

python
[] save mask/ bg removed version for all 1000 images
[] find way to compare, group, cluster, etc. the forms

---

things to consider

- disregard "fragment" pieces? ex - title: "Ceramic Fragment",
