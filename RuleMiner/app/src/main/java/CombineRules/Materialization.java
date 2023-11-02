package CombineRules;

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

import java.io.File;
import java.util.*;

public class Materialization {

	public static void main(String[] args) throws Exception {
		String[] datasets = new String[]{"WN18"};
		String[] models = new String[]{"ComplEx", "TransE", "TuckER"};


		for(String dataset_name: datasets){
			for(String model_name: models){
				System.out.println("Dataset name:" + dataset_name + "\tModel name:" + model_name);
				String neo4jFolder = "D:\\PhD\\Work\\EmbeddingInterpretibility\\RuleMiner\\db\\" + dataset_name + "_" + model_name + "_mispredicted\\";
				File folder = new File(neo4jFolder);
				if (folder.exists())
					MoreFiles.deleteRecursively(folder.toPath(), RecursiveDeleteOption.ALLOW_INSECURE);

				// 7(?a,?b)  => 8(?a,?b), Functional variable b
				// Support Query: MATCH (a)-[:7]->(b) MATCH (a)-[:8]->(b) RETURN a,b
				// PCA Query: MATCH (a)-[:7]->(b) MATCH ()-[:8]->(b) RETURN a,b

				String materialization_folder = "D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Results\\Materializations\\" + dataset_name + "\\" + model_name + "_mispredicted.tsv";
				Scanner sc = new Scanner(new File(materialization_folder));

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
		}


	}

}
