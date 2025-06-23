import { Card, CardContent } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useState, useEffect } from "react";

export default function Dashboard() {
  // Sample Data (Replace with API call to fetch real insights)
  const [data, setData] = useState([
    { name: "Accuracy", value: 92 },
    { name: "Precision", value: 88 },
    { name: "Recall", value: 85 },
    { name: "F1 Score", value: 86 },
  ]);

  return (
    <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Key Metrics Card */}
      <Card>
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold">ML Model Insights</h2>
          <p className="text-gray-500">Latest model performance metrics</p>
        </CardContent>
      </Card>
      
      {/* Chart for Model Performance */}
      <Card>
        <CardContent className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="value" fill="#4F46E5" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
