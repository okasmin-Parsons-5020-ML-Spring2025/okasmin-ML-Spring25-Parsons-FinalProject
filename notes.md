api documentation: https://metmuseum.github.io/

example call last objects: https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=%22vessel%22

example call single object: https://collectionapi.metmuseum.org/public/collection/v1/objects/479496

---

js
[] function to download image given object id - use primaryImageSmall field
[] download ~10 images to start - save to "data/initial_test/original" - name "objectId_original"

python
[] save copy of all images b&w - probably up the contrast
[] explore possible models
[] save copy of each image with background removed to "data/initial_test/bw" - name "objectId_bw"

---

js
[] download 1000 images
[] save data for 1000 objects to df / json

python
[] repeat process above for all 1000 images

---
