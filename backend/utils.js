export const testOutputFolder = "../public/data/initial_test/original";
export const testObjectId = "479496";

export const testObjectIds = [];

//API notes

// const baseUrl = "https://collectionapi.metmuseum.org/public/collection/v1/";

//https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=Auguste Renoir

export const createObjectUrl = (id) => {
  return `https://collectionapi.metmuseum.org/public/collection/v1/objects/${id}`;
};
