import axios from "axios";
import fs from "fs";
import path from "path";

import { createObjectUrl, testOutputFolder, testObjectId } from "./utils.js";

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
 * function to append objectId and imageUrl to json file
 */
const appendImageEntry = ({ objectId, imageUrl, filePath }) => {
  fs.readFile(filePath, "utf8", (err, data) => {
    let jsonArray;

    if (!err && data) {
      try {
        jsonArray = JSON.parse(data);
        if (!jsonArray) {
          jsonArray = [];
        }
      } catch (parseErr) {
        console.error("Error parsing JSON:", parseErr);
        return;
      }
    }

    // Add new entry
    jsonArray.push({ objectId, imageUrl });

    fs.writeFile(filePath, JSON.stringify(jsonArray, null, 2), (err) => {
      if (err) {
        console.error("Error writing JSON:", err);
      } else {
        // console.log(`Saved entry for ${objectId}`);
      }
    });
  });
};

/**
 * function to fetch object data json
 */
export const fetchObjectData = async ({ objectId, imageFolder, filePath }) => {
  if (!objectId || !imageFolder) {
    console.log("missing object id or imageFolder path");
    return;
  }

  try {
    const url = createObjectUrl(objectId);
    const { data } = await axios.get(url);

    const imageUrl = data.primaryImageSmall || data.primaryImage;

    if (!imageUrl) {
      console.log(`No image found for object ${objectId}`);
      return;
    }

    // save image URL to file - needed for BRIA model
    // console.log({ imageUrl });
    appendImageEntry({ objectId, imageUrl, filePath });

    // save original image to folder
    // fetchImage({ objectId, imageFolder, imageUrl });

    // TODO - handle other object data here in future
  } catch (err) {
    console.log("!!!error!!! ");
    console.error(err);
  }
};

/**
 * test call
 */
// fetchObjectData({ objectId: testObjectId, imageFolder: testOutputFolder });
