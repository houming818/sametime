# C10 raw training and resume smoke

Tasks 43 and 44 ran on `io`. The second task loaded the first task's remote
checkpoint at step 10 / row 960 and continued to step 20 / row 1,920. The large
checkpoint remains on io; summary and trace are retained here.
