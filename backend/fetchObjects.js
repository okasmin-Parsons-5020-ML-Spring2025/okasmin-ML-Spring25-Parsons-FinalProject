import axios from "axios";
import fs from "fs";
import { objectIdsFile } from "./utils.js";

/**
 * function to save array of object ids to json file
 */
const saveObjectIds = ({ ids, filePath }) => {
  fs.writeFile(filePath, JSON.stringify(ids, null, 2), (err) => {
    if (err) {
      console.error("Error writing JSON:", err);
    } else {
      console.log(`Saved ids`);
    }
  });
};

/**
 * function to fetch all object ids and save to json
 *  only need to run this once if allObjectIds.json is empty!!
 */
const getObjectIds = async () => {
  const url =
    "https://collectionapi.metmuseum.org/public/collection/v1/search?material=Ceramics&hasImages=true&isOnView=true&isPublicDomain=true&q=vase";

  try {
    const res = await axios.get(url);
    const total = res.data.total;
    const objectIds = res.data.objectIDs;

    console.log(total);
    console.log(objectIds);

    if (total && total > 0 && objectIds && objectIds.length > 0) {
      saveObjectIds({ ids: objectIds, filePath: objectIdsFile });
    }
  } catch (err) {
    console.log("!!!error!!! ");
    console.error(err);
  }
};
await getObjectIds();
