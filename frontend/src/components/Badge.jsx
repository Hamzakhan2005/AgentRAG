export default function Badge({ type, label }) {
  return <span className={`badge ${type}`}>{label}</span>;
}
