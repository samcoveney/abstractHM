# TODO

## emulator class
* [ ] 13-07-2018 : base class requires an global -> local dictionary to relate inputs of Tests to inputs dimensions of the emulator
* [ ] 13-07-2018 : need to fix several minmax issues across the board
* [X] 16-07-2018 : might be sensible to separate NROY from TESTS so we don't have multiple copies of data (same with corresponding I data)
* [X] 16-07-2018 : for simImp, could get around issue of points order by simply using calcImp with those points set as TESTS? Still lasting issue with possible difference in data order by user
* [ ] 17-07-2018 : fix how maxno is used in plot
* [ ] 17-07-2018 : float16 is really quite inaccurate, so perhaps user should specify storage type
