HierarchicalClustering.

This README was added for v6, aiming to fix issues with running with Java 7 on a Mac.

Note that we have incomplete source in SVN HEAD, both here and in the common directory.  
The code base used to reside under common only, but unfortunately that has changed over
time and the necessary code is not there.  The most recent version of the code can be 
found by going through the SVN Tags, combined with what is found here in the src dir.

Treat the code in 'src' as the most recent versions of those particular classes.  For
code that is not there, start by looking in the SVN Tags for previous releases.

It would probably be possible to bring together a full set of required files but it would 
require an effort of combing through SVN that may not be justified.  Instead, we are taking
the approach of just recovering individual files as necessary.  Doing otherwise brings the
risk of reintroducing old bugs back into the module.

Notes for v6:
- Made a local copy of the org.broadinstitute.gui.OS class.  This came
  out of 'common' rather than from any SVN Tag.
  Made changes to use Ant to detect the platform rather than relying
  on our old custom detection methods.  This addressed issues launching on a Mac with Java 7.
