import { fetchObjectData } from "./fetchObjectData.js";
import { testObjectIds, testOutputFolder } from "./utils.js";
// import pLimit from "p-limit";

// const concurrencyLimit = 10;
// const limit = pLimit(concurrencyLimit);

/**
 * function to loop through array of objectIds and fetch their data
 * batch fetching using p-limit
 */
