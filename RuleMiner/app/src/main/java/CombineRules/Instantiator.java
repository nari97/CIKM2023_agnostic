package CombineRules;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import org.neo4j.batchinsert.BatchInserter;
import org.neo4j.batchinsert.BatchInserters;
import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.dbms.api.DatabaseManagementServiceBuilder;
import org.neo4j.graphdb.*;
import org.neo4j.io.layout.DatabaseLayout;

public class Instantiator {

    public Instantiator(){

    }

    public static void createNewZip(String zipFileName) {
        ZipOutputStream zipOutputStream = null;
        try {

            // Delete existing zip if it exists
            File zipFile = new File(zipFileName);
            if (zipFile.exists()) {
                Files.delete(Paths.get(zipFileName));
            }

            // Create new zip file
            zipOutputStream = new ZipOutputStream(new FileOutputStream(zipFileName));


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void write_instantiations(BufferedWriter writer, Rule rule, GraphDatabaseService db) throws Exception {
        String query = "";
        Result res = null;

        //Find non functional variable

        String nonFuncVar = rule.functional_variable.equals("a")?"b":"a";

        //Build query
        for(Atom atom: rule.body_atoms)
            query += " MATCH " + atom.neo4j_print();
        
        Transaction tx = db.beginTx();
        
        System.out.println(new Date() + " -- Running body query");

        //Define hashmap
        Map<Long, Set<Long>> bodyPairs = new HashMap<>();
        try {
        	res = tx.execute(query + " RETURN id(a) AS a, id(b) AS b");
            while (res.hasNext()) {
                Map<String, Object> row = res.next();

                // Get functional and non functional variable
                long fv = (long) row.get(rule.functional_variable), nfv = (long) row.get(nonFuncVar);

                // If nv does not exist, add nv using fv as key
                if (!bodyPairs.containsKey(fv))
                	bodyPairs.put(fv, new HashSet<>());
                
                bodyPairs.get(fv).add(nfv);
            }
            res.close();
        } catch(Exception e){
            e.printStackTrace();
        }
        
        System.out.println(new Date() + " -- Running head query");

        //Define hash set for support and pca
        Set<String> support = new HashSet<>(), pca = new HashSet<>();
        int totalHeads = 0;

        // Define set for functional variables
        Set<Long> allFVs = new HashSet<>();

        // Define query for head atom
        res = tx.execute("MATCH " + rule.head_atom.neo4j_print() + " RETURN id(a) AS a, id(b) AS b");
        while (res.hasNext()) {
        	Map<String, Object> row = res.next();

            //Get fv and nfv
        	long fv = (long) row.get(rule.functional_variable), nfv = (long) row.get(nonFuncVar);

            // Add fv to fv set
        	allFVs.add(fv);

            // If bodypairs contains fv and the value is the nfv, add to support
            if (bodyPairs.containsKey(fv) && bodyPairs.get(fv).contains(nfv))
            	support.add(fv + "," + nfv);
            
            totalHeads++;
        }
        res.close();
        
        System.out.println(new Date() + " -- Computing PCA");

        // Compute pca
        // Iterate through all fvs
        for (Long fv : allFVs)
            // If bodyPairs contains the fv
        	if (bodyPairs.containsKey(fv))
                // For all values in key of body pairs, add to pca
        		for (Long other : bodyPairs.get(fv))
        			pca.add(fv + "," + other);
        
        System.out.println("Body pairs size: " + bodyPairs.size());
        System.out.println("Head size: " + totalHeads);
        System.out.println("Support size: " + support.size());
        System.out.println("PCA size: " + pca.size());

        // make pca and hc
        System.out.println("Head coverage: " + (1.0*support.size())/totalHeads);
        System.out.println("PCA confidence: " + (1.0*support.size())/pca.size());
        
        write_to_file(writer, support, pca, rule);
        tx.close();
    }

    public static void write_to_file(BufferedWriter writer, Set<String> support, Set<String> pca, Rule rule) throws IOException {

        writer.write(rule.id_print()+","+rule.head_coverage+","+rule.pca_confidence);
        writer.newLine();
        writer.flush();

        writer.write("" + support.size());
        writer.newLine();
        writer.flush();

        for(String res: support) {
            writer.write(res);
            writer.newLine();
            writer.flush();
        }

        writer.write("" + pca.size());
        writer.newLine();
        writer.flush();

        for(String res: pca) {
            writer.write(res);
            writer.newLine();
            writer.flush();
        }
    }

    public static void create_database(String database_path, String materialization_file_name) throws IOException {

        File folder = new File(database_path);
        if (folder.exists())
            MoreFiles.deleteRecursively(folder.toPath(), RecursiveDeleteOption.ALLOW_INSECURE);

        Scanner sc = new Scanner(new File(materialization_file_name));

        BatchInserter inserter = BatchInserters.inserter(DatabaseLayout.of(
                Config.newBuilder().set(GraphDatabaseSettings.neo4j_home, folder.toPath()).build()));

        int ctr = 0;
        while (sc.hasNextLine()) {
            String line = sc.nextLine();

            if (ctr%1000000 == 0){
                System.out.println("\tInserted rows:" + ctr);
            }
            ctr+=1;

            if(line.equals(""))
                continue;
            String[] spo = line.split("\t");
            long s = Long.valueOf(spo[0]), o = Long.valueOf(spo[2]);

            if (!inserter.nodeExists(s))
                inserter.createNode(s, new HashMap<>(), Label.label("Node"));
            if (!inserter.nodeExists(o))
                inserter.createNode(o, new HashMap<>(), Label.label("Node"));
            inserter.createRelationship(s, o, RelationshipType.withName(spo[1]), new HashMap<>());
        }

        inserter.shutdown();

        sc.close();
    }


    public static void main(String[] args) throws IOException {
        String dataset_name = args[0];
        String model_name = args[1];
        String folder_to_rules = args[2];
        String folder_to_db = args[3];
        String folder_to_instantiations = args[4];
        String folder_to_materialization = args[5];
        int mat_type = Integer.parseInt(args[6]);
        double beta = Double.parseDouble(args[7]);
        String predicates_to_parse = args[8];

        String file_name = dataset_name + "/" + model_name + "/" + dataset_name + "_" + model_name + "_" + "mat" + "_" + mat_type + ".tsv";
        System.out.println(new Date() + " -- Model:" + model_name + "; Dataset:" + dataset_name + "; Type:"+mat_type);
        String rule_filename = folder_to_rules + "/" + file_name.replace(".tsv", "") + "_rules.tsv";

        RuleParser rp = new RuleParser(rule_filename, null, model_name, dataset_name, "\t");
        rp.parse_rules_from_file(beta);

        String neo4jFolder = folder_to_db + "/" + file_name.replace(".tsv", "") + "/";

        create_database(neo4jFolder, folder_to_materialization + "/" + file_name);

        File folder = new File(neo4jFolder);
        DatabaseManagementService service = new DatabaseManagementServiceBuilder(folder.toPath()).
                setConfig(GraphDatabaseSettings.keep_logical_logs, "false").
                setConfig(GraphDatabaseSettings.preallocate_logical_logs, false).
                setConfig(GraphDatabaseSettings.keep_logical_logs, "false").
                setConfig(GraphDatabaseSettings.preallocate_logical_logs, false).build();

        GraphDatabaseService db = service.database("neo4j");
        System.out.println("Started neo4j");

        String resultFolderString = folder_to_instantiations + "/" + file_name.replace(".tsv", "") + "/";
        File result = new File(resultFolderString);
        if (!result.exists())
            result.mkdir();
        try {
            int ctr = 0;
            Set<String> predicates_to_instantiate = new HashSet<>();
            if (predicates_to_parse.length() == 0)
                predicates_to_instantiate = rp.rules_by_predicate.keySet();
            else{
                String[] splits = predicates_to_parse.split(",");
                predicates_to_instantiate.addAll(Arrays.asList(splits));
            }
            for (String predicate : predicates_to_instantiate) {
                System.out.println("Predicate: " + predicate + "\tPredicates left: " + (predicates_to_instantiate.size()-ctr));
                ctr+=1;
                File resultFolder = new File(resultFolderString + predicate + "/");
                if (!resultFolder.exists()) {
                    resultFolder.mkdir();
                }
                List<Rule> rules = rp.rules_by_predicate.get(predicate);

                for (int i = 0; i < rules.size() && i < 25; ++i) {
                    System.out.println("\t" + model_name + " " + dataset_name + " " + "Rule:" + i + "; " + rules.get(i).id_print());
                    String zipName = resultFolderString + predicate + "/" + "r" + i + ".zip";
                    createNewZip(zipName);
                    ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(zipName, true));

                    ZipEntry zipEntry = new ZipEntry("data.txt");
                    zipOutputStream.putNextEntry(zipEntry);

                    // Create buffered writer and write text to file
                    BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(zipOutputStream));
                    write_instantiations(bufferedWriter, rules.get(i), db);

                    zipOutputStream.closeEntry();
                    bufferedWriter.close();
                }
            }
            System.out.println("Finished instantiating");
        }catch (Exception e){
            e.printStackTrace();
            service.shutdown();
        }
        service.shutdown();

    }
}

