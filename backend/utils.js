// const baseUrl = "https://collectionapi.metmuseum.org/public/collection/v1/";

export const createObjectUrl = (id) => {
  return `https://collectionapi.metmuseum.org/public/collection/v1/objects/${id}`;
};

/**
 * REAL variables
 */
export const objectIdsFile = "public/data/allObjectIds.json";
export const objectIdsFile2 = "public/data/allObjectIds2.json";

/**
 * TEST variables
 */
export const testOutputFolder = "../public/data/initial_test/original";
export const testObjectId = "479496";

export const testObjectIds = ["477748", "447776", "48575", "771036", "548324"];
export const testUrlFile = "public/data/initial_test/imageUrls.json";

/**
 * selection of ceramic vases
 */

/**
 * https://www.metmuseum.org/art/collection/search/477748
https://www.metmuseum.org/art/collection/search/447776
https://www.metmuseum.org/art/collection/search/48575
https://www.metmuseum.org/art/collection/search/771036
https://www.metmuseum.org/art/collection/search/548324
 */
