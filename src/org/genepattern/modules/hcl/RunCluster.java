/*
 The Broad Institute
 SOFTWARE COPYRIGHT NOTICE AGREEMENT
 This software and its documentation are copyright (2003-2006) by the
 Broad Institute/Massachusetts Institute of Technology. All rights are
 reserved.

 This software is supplied without any warranty or guaranteed support
 whatsoever. Neither the Broad Institute nor MIT can be responsible for its
 use, misuse, or functionality.
 */

package org.genepattern.modules.hcl;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.tools.ant.Project;
import org.apache.tools.ant.Target;
import org.apache.tools.ant.taskdefs.Chmod;
import org.apache.tools.ant.taskdefs.Execute;
import org.genepattern.io.DatasetParser;
import org.genepattern.io.IOUtil;
import org.genepattern.io.stanford.StanfordTxtWriter;
import org.genepattern.matrix.Dataset;
import org.genepattern.module.AnalysisUtil;

public class RunCluster {

    public static void main(String[] args) {
        try {

            String libdir = args[0];
            boolean chmodFlag = true;
            String executable = null;
            if (System.getProperty("os.name").toLowerCase().startsWith(
                    "windows")) {
                executable = "cluster.exe";
                chmodFlag = false;
            } else if (System.getProperty("mrj.version") != null) {
                executable = "clusterMac";
            } else {
                executable = "clusterLinux";
            }
            if (chmodFlag) {
                MyChmod chmod = new MyChmod();
                chmod.setDir(new File(libdir));
                chmod.setIncludes(executable);
                chmod.setPerm("+x");
                chmod.execute();
            }
            args[0] = libdir + executable;

            String inputFileName = args[2];

            DatasetParser reader = AnalysisUtil
                    .getDatasetParser(inputFileName);

            Dataset dataset = AnalysisUtil.readDataset(
                    reader, inputFileName);

            StanfordTxtWriter writer = new StanfordTxtWriter();
            String txtFileName = IOUtil.getBaseFileName(inputFileName) + ".txt";
            BufferedOutputStream os = null;
            try {
                os = new BufferedOutputStream(new FileOutputStream(txtFileName));
                writer.write(dataset, os);
            } finally {
                if (os != null) {
                    os.close();
                }
                new File(txtFileName).deleteOnExit();
            }

            args[2] = txtFileName;
            List newArgs = new ArrayList(Arrays.asList(args));
            for (int i = 0; i < newArgs.size(); i++) {
                String arg = (String) newArgs.get(i);
                if (arg.equals("mean.row")) {
                    newArgs.remove(i); // median.row median.col
                    newArgs.add("-cg");
                    newArgs.add("a");
                    i--;
                } else if (arg.equals("median.row")) {
                    newArgs.remove(i);
                    newArgs.add("-cg");
                    newArgs.add("m");
                    i--;
                }
                if (arg.equals("mean.column")) {
                    newArgs.remove(i);
                    newArgs.add("-ca");
                    newArgs.add("a");
                    i--;
                } else if (arg.equals("median.column")) {
                    newArgs.remove(i);
                    newArgs.add("-ca");
                    newArgs.add("m");
                    i--;
                }
            }

            Execute execute = new Execute();
            String[] s = (String[]) newArgs.toArray(new String[0]);
            execute.setCommandline(s);
            execute.execute();
        } catch (Exception e) {
            e.printStackTrace();
            System.err
                    .println("An error occurred while running the algorithm.");
        }
    }

    static class MyChmod extends Chmod {
        public MyChmod() {
            project = new Project();
            project.init();
            taskType = "chmod";
            taskName = "chmod";
            target = new Target();
        }
    }

}