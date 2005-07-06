package org.genepattern.modules.hcl;

import org.apache.tools.ant.taskdefs.Chmod;
import org.apache.tools.ant.taskdefs.Execute;
import org.apache.tools.ant.Target;
import org.apache.tools.ant.Project;

import java.io.*;

import org.genepattern.io.expr.stanford.StanfordTxtWriter;
import org.genepattern.io.expr.IExpressionDataReader;
import org.genepattern.data.expr.ExpressionData;
import org.genepattern.module.AnalysisUtil;
import org.genepattern.ioutil.Util;

public class RunCluster {
    
    public static void main(String[] args) {
        try {
            String libdir = args[0];
            boolean chmodFlag = true;
            String executable = null;
            if(System.getProperty("os.name").toLowerCase().startsWith("windows")) {
                executable = "cluster.exe";
                chmodFlag = false;
            } else if(System.getProperty("mrj.version") != null) {
                executable = "clusterMac";
            } else {
                executable = "clusterLinux";
            }
            if(chmodFlag) {
                MyChmod chmod = new MyChmod();
                chmod.setDir(new File(libdir));
                chmod.setIncludes(executable);
                chmod.setPerm("+x");
                chmod.execute();
            }
            args[0] = libdir + executable;
            
            String inputFileName = args[2];
            
            IExpressionDataReader reader = AnalysisUtil
				.getExpressionReader(inputFileName);
            
            ExpressionData expressionData = AnalysisUtil.readExpressionData(reader,
                                                                            inputFileName);
           
            StanfordTxtWriter writer = new StanfordTxtWriter();
            String txtFileName = Util.getBaseFileName(inputFileName) + ".txt";
            BufferedOutputStream os = null;
            try {
                os = new BufferedOutputStream(new FileOutputStream(txtFileName));
                writer.write(expressionData, os);
            } finally {
                if(os!=null) {
                  os.close();  
                }
                new File(txtFileName).deleteOnExit();
            }
           
            args[2] = txtFileName;
            Execute execute = new Execute();
            execute.setCommandline(args);
            execute.execute();
        } catch(Exception e) {
            System.err.println("An error occurred while running the algorithm.");
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