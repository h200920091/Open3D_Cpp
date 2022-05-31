#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <fstream>
#include <iomanip>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace std;
using namespace open3d::visualization;

auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
auto rest = std::make_shared<open3d::geometry::PointCloud>();
map<int, Eigen::Vector4d> segmentModels;
map<int, shared_ptr<open3d::geometry::PointCloud>> segments;
vector<size_t> inliers;
int foundParallelNum = 0;

vector<double> segmentPlanesByRansac(string, int, double, bool, bool, bool, bool);
void xyzRgbToPly(string, bool);
double removeSegROI(shared_ptr<open3d::geometry::PointCloud>, double, bool, bool);
shared_ptr<open3d::geometry::PointCloud> removeOutlier(shared_ptr<open3d::geometry::PointCloud>, int, float, bool);
void createPoissonMeshSegment(string);

int main(int argc, char* argv[]) {
	utility::SetVerbosityLevel(utility::VerbosityLevel::Error);

	string filePath = "D:\\zhihao\\Project\\SSI\\CPP\\test\\build\\Release\\waferglued_2.txt";
	string outputPath = "D:\\zhihao\\Project\\SSI\\CPP\\test\\build\\Release\\waferglued_2_pcd.ply";
	string meshPath = "D:\\zhihao\\Project\\SSI\\CPP\\test\\build\\Release\\waferglued_2_mesh.ply";

	xyzRgbToPly(filePath, FALSE);

	auto planesThickness = segmentPlanesByRansac(outputPath, 6, 0.1, FALSE, TRUE, FALSE, FALSE);
	createPoissonMeshSegment(filePath);

	utility::LogInfo("End of the test.\n");
	return 0;
}

void xyzRgbToPly(string inputFile, bool csv) {
	string outputFile = inputFile.substr(0, inputFile.length() - 4) + +"_pcd.ply";
	auto pcdOption = open3d::io::ReadPointCloudOption::ReadPointCloudOption();

	if (!open3d::io::ReadPointCloudFromXYZRGB(inputFile, *cloudPtr, pcdOption)) {
		cout << "File Failed !!\n";
		return;
	}
	cout << "File Loaded !!\n" << endl;

	/* visualize PC */
	//open3d::visualization::DrawGeometries({ cloudPtr });

	auto search_param = geometry::KDTreeSearchParamKNN(250);
	cloudPtr->EstimateNormals(search_param);

	if (open3d::io::WritePointCloud(outputFile, *cloudPtr))
	{
		cout << "Saved as PLY !!\n\n";
	}

	if (!open3d::io::ReadPointCloud(outputFile, *cloudPtr)) { return; }
	cout << "PLY File Loaded !!\n\n";
}

vector<double> segmentPlanesByRansac(string inputFile, int maxPlaneIdx = 6, double parallelRatio = 0.1, bool downSample = FALSE, bool removeUselessPlane = FALSE, bool visualizeFoundPlane = FALSE, bool avoidVertical = FALSE) {
	auto maxBound = cloudPtr->GetMaxBound();
	auto minBound = cloudPtr->GetMinBound();
	double averageHeight = (maxBound[1] - minBound[1]) / maxPlaneIdx / 4;
	int idx = 0;
	rest = cloudPtr;
	foundParallelNum = 0;
	Eigen::Vector4d normalsToCmp;
	vector<double> planesThickness;

	while (foundParallelNum < maxPlaneIdx) {
		if (rest->points_.size() < 100)
			break;
		segmentModels[idx] = get<0>(rest->SegmentPlane(0.03, 3, 1000));
		inliers = get<1>(rest->SegmentPlane(0.03, 3, 1000));
		cout << "Plane " << idx << " equation: " << setprecision(3) << segmentModels[idx][0] << "x + "
			<< segmentModels[idx][1] << "y + " << segmentModels[idx][2] << "z + " << segmentModels[idx][3] << " = 0\n";

		// 先找是否為 Vertical plane
		if (avoidVertical && (abs(segmentModels[idx][0] - 0) < 0.1 && abs(segmentModels[idx][1] - 0) < 0.1 && abs(segmentModels[idx][2] - 0) < 0.1)) {
			cout << "Is vertical plane" << endl;
			if (removeUselessPlane) {
				auto notParallelPlane = rest->SelectByIndex(inliers);
				double thickness = removeSegROI(notParallelPlane, averageHeight, FALSE, visualizeFoundPlane);
			}
			continue;
		}
		// 先建立一個要比較的平面(之後取的面都要跟這個平行)
		else if (normalsToCmp.size() == 0) {
			normalsToCmp = segmentModels[idx];
		}

		// 平面若跟第一次找到的不是平行就先跳過
		else {
			Eigen::Vector2d dif(normalsToCmp[0] - segmentModels[idx][0], normalsToCmp[1] - segmentModels[idx][1]);
			if (max(dif[0], dif[1]) - min(dif[0], dif[1]) > parallelRatio) {
				cout << "not parallel\n";
				cout << dif << endl;
				if (removeUselessPlane) {
					auto notParallelPlane = rest->SelectByIndex(inliers);
					double thickness = removeSegROI(notParallelPlane, averageHeight, FALSE, visualizeFoundPlane);
				}
				continue;
			}
		}
		segments[idx] = rest->SelectByIndex(inliers);

		double thickness = removeSegROI(segments[idx], averageHeight, FALSE, visualizeFoundPlane);
		planesThickness.push_back(thickness);
		idx++;
		foundParallelNum++;
		cout << "pass " << idx << " / " << maxPlaneIdx << " done.\n\n";
	}

	// sorting

	return planesThickness;
}

double removeSegROI(shared_ptr<open3d::geometry::PointCloud> segment, double averageHeight, bool findThickness = TRUE, bool visualize = FALSE) {
	auto segBbox = segment->GetOrientedBoundingBox();
	Eigen::Vector3d color(1, 0, 0);
	segBbox.color_ = color;
	Eigen::Vector3d dilateExtent(segBbox.extent_[0] * 1.3, segBbox.extent_[1] * 1.3, averageHeight);
	auto toRemoveBbox = geometry::OrientedBoundingBox(segBbox.center_, segBbox.R_, dilateExtent);
	auto toRemoveIndexes = toRemoveBbox.GetPointIndicesWithinBoundingBox(rest->points_);
	double thickness = 0;
	if (findThickness) {
		// 用比較小的範圍找出跟厚度相關的 Bounding box
		Eigen::Vector3d erosionExtent(segBbox.extent_[0] / 2, segBbox.extent_[1] / 2, averageHeight);
		auto thicknessBbox = geometry::OrientedBoundingBox(segBbox.center_, segBbox.R_, erosionExtent);
		auto toRemoveSmallRoiIndexes = thicknessBbox.GetPointIndicesWithinBoundingBox(rest->points_);
		auto planeRegionPcd = rest->SelectByIndex(toRemoveSmallRoiIndexes, FALSE);
		bool removeOutlier_ = TRUE;
		if (removeOutlier_)
			planeRegionPcd = removeOutlier(planeRegionPcd, 40, 0.5, FALSE);
		auto planeRegionBbox = planeRegionPcd->GetOrientedBoundingBox();
		thickness = planeRegionBbox.extent_[2];
	}
	rest = rest->SelectByIndex(toRemoveIndexes, TRUE);
	return thickness;
}

shared_ptr<open3d::geometry::PointCloud> removeOutlier(shared_ptr<open3d::geometry::PointCloud> pcd, int nbNeighbors = 20, float stdRatio = 2.0, bool display = FALSE) {
	auto ind = get<1>(pcd->RemoveStatisticalOutliers(nbNeighbors, stdRatio));
	auto inlierCloud = pcd->SelectByIndex(ind);
	return inlierCloud;
}


void createPoissonMeshSegment(string filePath) {
	string outputFile = filePath.substr(0, filePath.length() - 4) + +"_mesh.ply";
	int segmentsCount = segments.size();
	cout << "Testing IO for meshes ...\n";
	auto currentMesh = geometry::TriangleMesh();
	auto& app = gui::Application::GetInstance();
	app.Initialize();
	auto vis = std::make_shared<visualizer::O3DVisualizer>("Pcd and Mesh Visualization", 1024, 768);
	vis->ShowSettings(TRUE);
	vis->ShowSkybox(FALSE);

	for (int i = 0; i < segmentsCount; i++) {
		auto poissonMesh = get<0>(geometry::TriangleMesh::CreateFromPointCloudPoisson(*segments[i], 8, 0, 1.1, FALSE));
		auto bbox = segments[i]->GetAxisAlignedBoundingBox();
		auto pMeshCrop = poissonMesh->Crop(bbox);
		string geometryName = "Mesh" + to_string(i);
		string pcdName = "Pcd" + to_string(i);
		auto maxPt = segments[i]->GetMinBound();
		string planeIndex = "Plane " + to_string(i);
		const char* temp = planeIndex.c_str();
		currentMesh += *pMeshCrop;
		vis->Add3DLabel(Eigen::Vector3f(maxPt[0], maxPt[1], maxPt[2]), temp);
		vis->AddGeometry(pcdName, segments[i], nullptr, "PCD");
		vis->AddGeometry(geometryName, pMeshCrop, nullptr, "Mesh");
	}
	open3d::io::WriteTriangleMesh(outputFile, currentMesh);

	vis->ResetCameraToDefault();
	app.AddWindow(vis);
	app.Run();

}