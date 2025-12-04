import React, { useEffect, useRef } from "react";

function MatchSummaryPrint({ schema, summaryData = [], buttonClass = "match-btn-purple" }) {
  const btnRef = useRef(null);

  useEffect(() => {
    // Attach native click event (bypasses React synthetic event)
    const btn = btnRef.current;
    if (!btn) return;

    const handleClick = () => {
      if (!summaryData || summaryData.length === 0) {
        alert("No data available for printing.");
        return;
      }

      // ✅ Open new tab immediately
      const win = window.open("", "_blank", "width=1024,height=800");
      if (!win) {
        alert("Popup blocked! Please allow pop-ups for this site.");
        return;
      }

      // Build HTML
      const style = `
        <style>
          body { font-family: Arial, sans-serif; padding: 28px; }
          h1 { text-align: center; margin: 0; font-size: 22px; }
          h3 { text-align: center; margin: 6px 0 20px 0; font-size: 15px; color: #333; }
          table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
          th, td { border: 1px solid #999; padding: 6px; text-align: center; }
          th { background: #e3e3e3; font-weight: 600; }
          tr:nth-child(even) { background: #fafafa; }
        </style>
      `;

      const header = `
        <div style="text-align:center; margin-bottom:20px;">
          <h1><b>MATCH SUMMARY REPORT</b></h1>
          <h3>Schema: ${schema || "N/A"}</h3>
        </div>
      `;

      const rows = summaryData
        .map(
          (r) => `
          <tr>
            <td>${r.code || ""}</td>
            <td>${r.barangay || ""}</td>
            <td>${r.gdb || ""}</td>
            <td>${r.rdb || ""}</td>
            <td style="color:${
              r.rdb_minus_gdb > 0
                ? "green"
                : r.rdb_minus_gdb < 0
                ? "red"
                : "black"
            };">${r.rdb_minus_gdb ?? ""}</td>
            <td>${r.total_match ?? ""}</td>
            <td>${r.rdb_over_gdb?.toFixed(2) ?? ""}%</td>
            <td>${r.match_over_gdb?.toFixed(2) ?? ""}%</td>
            <td>${r.match_over_rdb?.toFixed(2) ?? ""}%</td>
          </tr>`
        )
        .join("");

      // ✅ Write synchronously before JS event stack clears
      win.document.write(`
        <html>
          <head><title>Match Summary Report</title>${style}</head>
          <body>
            ${header}
            <table>
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Barangay</th>
                  <th>GDB</th>
                  <th>RDB</th>
                  <th>RDB - GDB</th>
                  <th>Total Match</th>
                  <th>RDB / GDB</th>
                  <th>Total Match / GDB</th>
                  <th>Total Match / RDB</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
            <script>window.onload = () => { window.focus(); window.print(); }</script>
          </body>
        </html>
      `);
      win.document.close();
    };

    // Add native listener
    btn.addEventListener("click", handleClick);
    return () => btn.removeEventListener("click", handleClick);
  }, [schema, summaryData]);

  return (
    <button ref={btnRef} className={buttonClass} disabled={!summaryData?.length}>
      Print Preview
    </button>
  );
}

export default MatchSummaryPrint;
