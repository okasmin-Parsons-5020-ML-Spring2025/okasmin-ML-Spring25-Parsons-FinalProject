# okasmin-ML-Spring25-Parsons-FinalProject

## Idea 1 - Ceramic Vessel Forms in the Metropolitan Museum of Art Collection

### Concept

I’m interested in analyzing the shapes of ceramic vessels across different cultures and centuries to determine the most common forms, the least common forms, the overlap between cultures, perhaps an average of forms, etc. The Metropolitan Museum of Art collection includes a large selection of ceramics ranging from ancient to contemporary, and I think it could be a good sample for this data.

I started making ceramics a few years ago and am usually drawn to creating functional forms. Seeing ancient vases and bowls in museums that resemble the objects I make for everyday use fills me with awe. It’s a powerful reminder of the connection we share with past cultures—through the craft itself, the handmade nature of the work, and the consistency of human behavior.

Given infinite time, I think developing an interactive visualization tool to explore this aspect of the collection could be fun—perhaps even allowing users to import an image of their own work and see what objects it most closely resembles from the collection. As an end goal for this project, I would definitely like to include some sort of visual output, perhaps a generated image of the average form of each group, but I’m not entirely sure yet what that might look like.

### Data

The Metropolitan Museum of Art has an open API that allows you to search for groups of objects and get detailed information, including an image link, based on the object id.

The first step will be determining the correct query terms to select all ceramic vessels and then iterating through the list of object IDs returned to download an image for each object. The images can be named using the object ID, and then I can save the other data regarding the object in a table to be referenced by that object ID.

An initial search for “vessel” for objects with images returns 44,307 objects, and “ceramic” returns 29,466 objects, so I may need to apply further filters or perhaps gather a random sample from each museum department.

`https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=%22vessel%22`

`https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=%22ceramic%22`

I’m envisioning training a model to select the primary form from an image (ignoring shadows, and tossing out any images that include multiple objects) and then grouping that form with others that are similar.

The images are all relatively consistent in the way they’re photographed, though some are in B&W and some are in color. Since this model will focus on form, I’ll likely convert all images to B&W.

I’m not sure how prescriptive I will have to be to determine what “similar” means—it could be interesting to fiddle with a parameter that determines *how* similar forms must be to be grouped together, or perhaps to tell the model how many groups it should return. These types of adjustments should hopefully present different patterns and trends.

I might consider putting together a training dataset with a limited number of images that will be relatively easy for me to evaluate visually as I fine-tune the model.

#### Sample Images

<img width="332" alt="Screenshot 2025-04-01 at 2 08 07 PM" src="https://github.com/user-attachments/assets/ab515915-b8c8-4725-86ab-3cb7d9511983" />
<img width="334" alt="Screenshot 2025-04-01 at 2 06 16 PM" src="https://github.com/user-attachments/assets/9c34b40c-892d-48be-8870-e9dd4aed46da" />
<img width="338" alt="Screenshot 2025-04-01 at 2 06 54 PM" src="https://github.com/user-attachments/assets/f2a85a60-d518-4bde-af44-a8ff2baa0bc7" />
<img width="324" alt="Screenshot 2025-04-01 at 2 05 20 PM" src="https://github.com/user-attachments/assets/55decde5-0860-41d1-a66b-612790821e1e" />
<img width="321" alt="Screenshot 2025-04-01 at 2 05 46 PM" src="https://github.com/user-attachments/assets/f6bed73a-ef25-4d9e-9693-467a6c41c44b" />


Sources:

https://www.metmuseum.org/art/collection/search/20025

https://www.metmuseum.org/art/collection/search/206452

https://www.metmuseum.org/art/collection/search/62658

https://www.metmuseum.org/art/collection/search/551842

https://www.metmuseum.org/art/collection/search/326186
