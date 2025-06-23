// index.js
import React, { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";

const Home = () => {
    const [file, setFile] = useState(null);
    const [dropColumns, setDropColumns] = useState("");
    const [targetColumn, setTargetColumn] = useState("");
    const [featureColumn, setFeatureColumn] = useState("");
    const [task, setTask] = useState("regression");
    const [result, setResult] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async () => {
        if (!file) {
            alert("Please upload a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("drop_columns", dropColumns);

        if (task !== "cluster" && task !== "visualize") {
            formData.append("target_column", targetColumn);
        }
        if (task !== "visualize") {
            formData.append("feature_column", featureColumn);
        }

        try {
            const storeResponse = await axios.post("http://127.0.0.1:8000/upload_csv/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            const taskResponse = await axios.post(`http://127.0.0.1:8000/${task}/`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setResult(taskResponse.data);
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred. Please check the console for details.");
        }
    };

    return (
        <div className="flex min-h-screen bg-gradient-to-br from-gray-100 to-gray-300">
            {/* Sidebar */}
            <div className="w-64 bg-white shadow-lg p-6">
                <h1 className="text-2xl font-bold text-gray-800 mb-6">ML Insights</h1>
                <nav className="space-y-3">
                    <Button className="w-full justify-start" variant="ghost">
                        Dashboard
                    </Button>
                    <Button className="w-full justify-start" variant="ghost">
                        Regression
                    </Button>
                    <Button className="w-full justify-start" variant="ghost">
                        Classification
                    </Button>
                    <Button className="w-full justify-start" variant="ghost">
                        Clustering
                    </Button>
                    <Button className="w-full justify-start" variant="ghost">
                        Visualization
                    </Button>
                </nav>
            </div>

            {/* Main Content */}
            <div className="flex-1 p-8">
                <Card className="w-full max-w-2xl p-6 shadow-2xl bg-white rounded-2xl">
                    <CardContent>
                        <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">Upload CSV and Run Analysis</h2>

                        {/* File Upload */}
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="w-full p-2 border rounded-md mb-4 file:bg-blue-500 file:text-white file:px-4 file:py-2 file:rounded-md"
                        />

                        {/* Dropdown to select task */}
                        <Select value={task} onValueChange={setTask}>
                            <SelectTrigger className="w-full p-3 border rounded-md bg-gray-100">
                                <SelectValue placeholder="Select Task" />
                            </SelectTrigger>
                            <SelectContent className="bg-white shadow-md rounded-md">
                                <SelectItem value="regression">Regression</SelectItem>
                                <SelectItem value="classification">Classification</SelectItem>
                                <SelectItem value="cluster">Clustering</SelectItem>
                                <SelectItem value="visualize">Visualization</SelectItem>
                            </SelectContent>
                        </Select>

                        {/* Input fields */}
                        <div className="mt-4 space-y-3">
                            <input
                                type="text"
                                className="w-full p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-300"
                                placeholder="Columns to drop (comma-separated)"
                                value={dropColumns}
                                onChange={(e) => setDropColumns(e.target.value)}
                            />

                            {(task === "regression" || task === "classification") && (
                                <input
                                    type="text"
                                    className="w-full p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-300"
                                    placeholder="Target column"
                                    value={targetColumn}
                                    onChange={(e) => setTargetColumn(e.target.value)}
                                />
                            )}

                            {task !== "visualize" && (
                                <input
                                    type="text"
                                    className="w-full p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-300"
                                    placeholder="Impactful Features (comma-separated)"
                                    value={featureColumn}
                                    onChange={(e) => setFeatureColumn(e.target.value)}
                                />
                            )}
                        </div>

                        {/* Submit button */}
                        <Button className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg transition-all" onClick={handleSubmit}>
                            Run {task.charAt(0).toUpperCase() + task.slice(1)}
                        </Button>

                        {/* Display the result */}
                        {result && (
                            <div className="mt-6 p-4 border rounded-lg bg-gray-100">
                                <strong className="text-lg text-gray-800">Result:</strong>
                                <pre className="text-sm text-gray-600 mt-2">{JSON.stringify(result, null, 2)}</pre>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default Home;