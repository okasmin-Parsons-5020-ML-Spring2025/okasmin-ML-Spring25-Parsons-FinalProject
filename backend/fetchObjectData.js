import axios from "axios";
import fs from "fs";
import path from "path";

import { testOutputFolder, testObjectId } from "./utils.js";

const createUrl = (id) => {
  return `https://collectionapi.metmuseum.org/public/collection/v1/objects/${id}`;
};

/**
 * function to handle downloading & saving image from object data response
 */
const fetchImage = async ({ objectId, imageFolder, imageUrl }) => {
  try {
    if (!fs.existsSync(imageFolder)) {
      // create imageFolder if doesn't exist
      // fs.mkdirSync(imageFolder, { recursive: true });
      console.log(`imageFolder does not exist: ${imageFolder}`);
      return;
    }

    const fileName = `${objectId}.jpg`;
    const filePath = path.join(imageFolder, fileName);

    if (fs.existsSync(filePath)) {
      console.log(`skipping because "${fileName}" is already in "${imageFolder}"`);
      return;
    }

    console.log(`download image for object ${objectId} to ${filePath}`);

    const imageResponse = await axios({
      url: imageUrl,
      method: "GET",
      responseType: "stream",
    });

    const writer = fs.createWriteStream(filePath);
    imageResponse.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on("finish", () => {
        console.log(`saved: ${fileName}`);
        resolve();
      });
      writer.on("error", (err) => {
        console.error(`failed to save ${fileName}`, err);
        reject(err);
      });
    });
  } catch (err) {
    console.error(err);
  }
};

/**
 * function to fetch object data json
 */
export const fetchObjectData = async ({ objectId, imageFolder }) => {
  if (!objectId || !imageFolder) {
    console.log("missing object id or imageFolder path");
    return;
  }

  try {
    const url = createUrl(objectId);
    const { data } = await axios.get(url);

    const imageUrl = data.primaryImageSmall || data.primaryImage;

    if (!imageUrl) {
      console.log(`No image found for object ${objectId}`);
      return;
    }

    fetchImage({ objectId, imageFolder, imageUrl });

    // TODO - handle other object data here in future
  } catch (err) {
    console.log("!!!error!!! ");
    console.error(err);
  }
};

/**
 * test call
 */
fetchObjectData({ objectId: testObjectId, imageFolder: testOutputFolder });
