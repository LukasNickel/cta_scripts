# cta_scripts

All sorts of small scripts to work with CTA data

## merge_dl1.py
Merge any number of dl1 files to one bigger file
There are some caveats to remember:
- The first file acts as a base, later files get appended
- Everything in the /configuration group does not get appended.
This is done to avoid duplicated camera infos and such things.
You need to make sure the configurations are all the same
(layout, camera, ...) or later scripts might lead to wrong results.
- Sometimes table dtypes are not the same. This affects the parameters tables
(float32 vs float64) and the pointings tables (tels_with_trigger).
In these cases, the everything gets converted to the dtypes of the first table.
Should not matter for the parameters in most cases and the tels_with_trigger
column in the pointings table has no use anyway, because pointings are
used for multiple events. A warning is printed for each conversion.
